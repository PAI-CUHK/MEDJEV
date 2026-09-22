"""Paired schema-curriculum experiment: same initialization, rows, updates and CE."""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from .gpu import MODEL, REVISION, ClinicalDecisionModel, tokenize_rows, infer, save_model, load_model
from .data import load_split, dev_partition, audit, write_json
from .experiment import seed_all
from .metrics import metrics, probabilities, fit_temperature
from .schemas import SCHEMAS, BY_ID, draw_schema, schema_key


def train(args):
    out = Path(args.output)
    if out.exists():
        raise ValueError("Use a fresh run directory")
    out.mkdir(parents=True)
    seed_all(args.seed)
    torch.set_num_threads(8)
    rows = load_split(args.data,"train")
    selection, calibration = dev_partition(load_split(args.data,"dev"))
    write_json(out/"data_audit.json",audit(args.data))
    write_json(out/"schema_manifest.json",[asdict(s) for s in SCHEMAS])
    tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    base=AutoModel.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.bfloat16,attn_implementation="sdpa")
    backbone=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="FEATURE_EXTRACTION"))
    model=ClinicalDecisionModel(backbone,"dynamic").cuda()
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.01)
    meta={"model":MODEL,"revision":REVISION,"mode":"dynamic","max_length":args.max_length,
          "seed":args.seed,"curriculum":args.curriculum,"temperature":1.,"status":"training",
          "selection":"minimum canonical development NLL", "epochs":args.epochs,
          "batch_size":args.batch_size,"lr":2e-4,"lora_rank":16,
          "implementation":"v2 paired schema experiment; no head or optimizer changes"}
    write_json(out/"run_config.json",meta)
    order_rng=np.random.default_rng(args.seed)
    schema_rng=np.random.default_rng(args.seed+10000)
    perm_rng=np.random.default_rng(args.seed+20000)
    history=[]
    counts=Counter()
    total_paths=0
    best=float("inf")
    started=time.perf_counter()
    for epoch in range(args.epochs):
        order=order_rng.permutation(len(rows))
        model.train()
        loss_sum=0.
        for step, start in enumerate(range(0,len(rows),args.batch_size)):
            # Identical row order in paired arms; schema sampling has a separate RNG.
            chunk=[rows[i] for i in order[start:start+args.batch_size]]
            schema=draw_schema(schema_rng,args.curriculum)
            perm=perm_rng.permutation(len(schema.candidates))
            inverse=np.argsort(perm)
            descriptions=[schema.candidates[i] for i in perm]
            batch=tokenize_rows(chunk,tokenizer,"dynamic",args.max_length,descriptions,schema.question)
            labels=torch.tensor([inverse[schema.mapping[r["label"]]] for r in chunk],device="cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=model(batch,len(descriptions))
                loss=torch.nn.functional.cross_entropy(logits.float(),labels)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step()
            loss_sum+=loss.item()*len(chunk)
            counts[schema.id]+=len(chunk)
            total_paths+=len(chunk)*len(descriptions)
            if step%150==0:
                print(f"seed={args.seed} curriculum={args.curriculum} epoch={epoch+1} step={step} loss={loss.item():.4f}",flush=True)
        logits=infer(model,tokenizer,selection,args)
        dev=metrics(np.array([r["label"] for r in selection]),probabilities(logits))
        record={"epoch":epoch+1,"train_loss":loss_sum/len(rows),"selection":dev,
                "elapsed_seconds":time.perf_counter()-started}
        history.append(record)
        write_json(out/"history.json",history)
        print(json.dumps({"epoch":epoch+1,"dev_accuracy":dev["accuracy"],"dev_nll":dev["nll"]}),flush=True)
        if dev["nll"]<best:
            best=dev["nll"]
            meta["best_epoch"]=epoch+1
            save_model(model,tokenizer,out,meta)
    del optim,model,backbone,base
    gc.collect()
    torch.cuda.empty_cache()
    model,tokenizer,meta=load_model(out)
    meta["schema_temperatures"]={}
    # Fit temperatures only for exact deployment schemas, never held-out wordings.
    for name in ["nli","support","refute","determined"]:
        s=BY_ID[name]
        z=infer(model,tokenizer,calibration,args,list(s.candidates),s.question)
        y=np.array([s.mapping[r["label"]] for r in calibration])
        t=fit_temperature(z,y)
        meta["schema_temperatures"][schema_key(s.question,s.candidates)]=t
        if name=="nli":
            meta["temperature"]=t
    meta.update(status="trained",training_seconds=time.perf_counter()-started,
                schema_exposure=dict(counts),candidate_paths=total_paths,test_evaluated=False)
    write_json(out/"metadata.json",meta)
    print("Training and exact-schema calibration finished. No test evaluation.",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data")
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=24)
    p.add_argument("--max-length",type=int,default=768)
    p.add_argument("--curriculum",action="store_true")
    train(p.parse_args())
