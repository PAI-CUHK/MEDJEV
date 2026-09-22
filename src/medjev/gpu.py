"""GPU LoRA experiments: dynamic candidate scoring vs matched fixed classifier.

Original implementation inspired by NanoJev candidate-path scoring.
Not a reproduction of TypeSafe RLCD. Candidate paths re-encode the evidence.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from .data import load_split, dev_partition, LABELS, DESCRIPTIONS, QUESTION, audit, write_json
from .metrics import metrics, probabilities, fit_temperature
from .experiment import seed_all

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def render(row, candidate=None, question=QUESTION):
    text = (f"Clinical evidence:\n{row['premise']}\n\n"
            f"Statement:\n{row['hypothesis']}\n\nQuestion: {question}\n")
    return text + (f"Candidate meaning: {candidate}\nAssess this candidate:" if candidate is not None
                   else "Classify the evidence relationship:")


class ClinicalDecisionModel(nn.Module):
    def __init__(self, backbone, mode, n_labels=3, interaction="independent"):
        super().__init__()
        self.backbone, self.mode, self.n_labels = backbone, mode, n_labels
        self.interaction = interaction
        h = getattr(backbone.config, "hidden_size", None) or backbone.config.text_config.hidden_size
        self.cross = nn.MultiheadAttention(h, num_heads=1, batch_first=True).float() if interaction == "cross_attention" and mode == "dynamic" else None
        self.set_proj = nn.Sequential(nn.LayerNorm(h * 2), nn.Linear(h * 2, h), nn.GELU()).float() if interaction == "set_attention" and mode == "dynamic" else None
        self.head = nn.Linear(h * (2 if interaction == "pairwise_set" and mode == "dynamic" else 1),
                              1 if mode=="dynamic" else n_labels).float()

    def forward(self, batch, n_candidates=3):
        states = self.backbone(**batch, use_cache=False).last_hidden_state
        last = batch["attention_mask"].sum(1)-1
        pooled = states[torch.arange(states.shape[0], device=states.device), last].float()
        if self.mode == "dynamic" and self.interaction == "pairwise_set":
            if pooled.shape[0] % n_candidates != 0:
                raise ValueError("dynamic candidate rows are not divisible by n_candidates")
            grouped = pooled.reshape(-1, n_candidates, pooled.shape[-1])
            context = grouped.mean(1, keepdim=True).expand_as(grouped)
            pooled = torch.cat([grouped, context], dim=-1).reshape(-1, pooled.shape[-1] * 2)
        elif self.mode == "dynamic" and self.interaction == "cross_attention":
            if pooled.shape[0] % n_candidates != 0:
                raise ValueError("dynamic candidate rows are not divisible by n_candidates")
            grouped = pooled.reshape(-1, n_candidates, pooled.shape[-1])
            attended, _ = self.cross(grouped, grouped, grouped, need_weights=False)
            pooled = (grouped + attended).reshape(-1, pooled.shape[-1])
        elif self.mode == "dynamic" and self.interaction == "set_attention":
            if pooled.shape[0] % n_candidates != 0:
                raise ValueError("dynamic candidate rows are not divisible by n_candidates")
            grouped = pooled.reshape(-1, n_candidates, pooled.shape[-1])
            # Symmetric set context: each candidate receives the same aggregate
            # of all candidate representations, then a shared projection scores it.
            mean_ctx = grouped.mean(1, keepdim=True).expand_as(grouped)
            max_ctx = grouped.max(1, keepdim=True).values.expand_as(grouped)
            contextual = self.set_proj(torch.cat([grouped, mean_ctx + max_ctx], dim=-1))
            # Residual preservation prevents the symmetric branch from collapsing
            # all candidates to one nearly uniform representation.
            pooled = (grouped + contextual).reshape(-1, pooled.shape[-1])
        scores = self.head(pooled)
        if self.mode=="dynamic":
            scores = scores.reshape(-1, n_candidates)
            # Optional set normalization: subtract a symmetric group statistic
            # before softmax, making the candidate score representation invariant
            # to the ordering of candidate rows.
            if self.interaction == "set_centered":
                scores = scores - scores.mean(dim=1, keepdim=True)
            return scores
        return scores


def candidate_text(description, index, template_version="legacy"):
    text = str(description)
    if text.startswith("Candidate ID:"):
        return text
    if template_version == "aligned_v2":
        return f"Candidate ID: {index}\n{text}"
    if template_version != "legacy":
        raise ValueError(f"Unknown candidate template: {template_version}")
    return f"Candidate ID: {index}\nCandidate meaning: {text}"


def tokenize_rows(rows, tokenizer, mode, max_length, descriptions=None, question=QUESTION):
    if mode == "dynamic":
        if descriptions is None:
            per_row = [r.get("candidate_meanings", DESCRIPTIONS) for r in rows]
        elif descriptions and isinstance(descriptions[0], (list, tuple)):
            per_row = descriptions
        else:
            per_row = [descriptions for _ in rows]
        # Explicit canonical candidate IDs make the dynamic path aware of identity,
        # rather than relying on the position of a permuted candidate row.
        texts = []
        for r, ds in zip(rows, per_row):
            for j, d in enumerate(ds):
                d = candidate_text(d, j, getattr(tokenizer, "medjev_template_version", "legacy"))
                texts.append(render(r, d, question))
    else:
        texts = [render(r, question=question) for r in rows]
    allow_truncation = os.environ.get("MEDJEV_ALLOW_TRUNCATION", "0") == "1"
    encoded = tokenizer(texts, truncation=allow_truncation, max_length=max_length, add_special_tokens=True)
    # Preserve complete clinical evidence: reject too-long samples rather than silently truncating.
    if (not allow_truncation) and any(len(ids)>max_length for ids in encoded["input_ids"]):
        longest = max(map(len, encoded["input_ids"]))
        raise ValueError(f"Input length {longest} exceeds configured {max_length}; increase max_length")
    return tokenizer.pad(encoded, padding=True, return_tensors="pt").to("cuda")


@torch.inference_mode()
def infer(model, tokenizer, rows, args, descriptions=None, question=QUESTION):
    model.eval()
    outputs = []
    for i in range(0, len(rows), args.batch_size):
        per_record = descriptions is not None and len(descriptions) > 0 and isinstance(descriptions[0], (list, tuple))
        if per_record and len(descriptions) != len(rows):
            raise ValueError("Per-record candidate descriptions must align with rows")
        batch_descriptions = descriptions[i:i+args.batch_size] if per_record else descriptions
        batch = tokenize_rows(rows[i:i+args.batch_size], tokenizer, model.mode, args.max_length, batch_descriptions, question)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if descriptions is not None and descriptions and isinstance(descriptions[0], (list, tuple)):
                n_candidates = len(descriptions[0])
            elif descriptions is not None:
                n_candidates = len(descriptions)
            else:
                n_candidates = len(rows[i].get("candidate_meanings", DESCRIPTIONS))
            outputs.append(model(batch, n_candidates).float().cpu().numpy())
    return np.concatenate(outputs)


def save_model(model, tokenizer, path, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path/"adapter")
    tokenizer.save_pretrained(path/"tokenizer")
    torch.save(model.head.state_dict(), path/"head.pt")
    if model.cross is not None:
        torch.save(model.cross.state_dict(), path/"cross.pt")
    if model.set_proj is not None:
        torch.save(model.set_proj.state_dict(), path/"set_proj.pt")
    write_json(path/"metadata.json", metadata)


def load_model(path, device="cuda"):
    path = Path(path)
    meta = json.loads((path/"metadata.json").read_text())
    base = AutoModel.from_pretrained(meta["model"], revision=meta["revision"],
                                    torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    backbone = PeftModel.from_pretrained(base, path/"adapter")
    model = ClinicalDecisionModel(backbone, meta["mode"], meta.get("n_labels", 3), meta.get("interaction", "independent")).to(device).eval()
    model.head.load_state_dict(torch.load(path/"head.pt", map_location=device, weights_only=True))
    if model.cross is not None:
        if not (path/"cross.pt").exists():
            raise ValueError("Incomplete cross_attention checkpoint: missing cross.pt; historical results require audit")
        model.cross.load_state_dict(torch.load(path/"cross.pt", map_location=device, weights_only=True))
    if model.set_proj is not None and (path/"set_proj.pt").exists():
        model.set_proj.load_state_dict(torch.load(path/"set_proj.pt", map_location=device, weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained(path/"tokenizer")
    tokenizer.medjev_template_version = meta.get("candidate_template", "legacy")
    return model, tokenizer, meta


def train(args):
    out = Path(args.output)
    if (out/"metadata.json").exists():
        raise ValueError("Output already contains a model; choose a new run directory")
    seed_all(args.seed)
    torch.set_num_threads(8)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"data_audit.json", audit(args.data, include_test=False))
    rows = load_split(args.data,"train")
    selection, calibration = dev_partition(load_split(args.data,"dev"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.medjev_template_version = args.candidate_template
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModel.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.bfloat16,
                                    attn_implementation="sdpa")
    base.config.use_cache = False
    backbone = get_peft_model(base, LoraConfig(r=args.lora_rank, lora_alpha=2*args.lora_rank,
        lora_dropout=.05, target_modules=["q_proj","k_proj","v_proj","o_proj"],
        task_type="FEATURE_EXTRACTION"))
    n_labels = len(LABELS)
    model = ClinicalDecisionModel(backbone,args.mode,n_labels,args.interaction).cuda()
    body_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for name, p in model.named_parameters()
                   if not name.startswith("backbone.") and p.requires_grad]
    optimized_ids = {id(p) for p in body_params + head_params}
    if optimized_ids != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError("Optimizer does not cover every trainable model parameter")
    head_lr = args.head_lr if args.head_lr is not None else args.lr
    optim = torch.optim.AdamW([{"params":body_params,"lr":args.lr},
                              {"params":head_params,"lr":head_lr}], weight_decay=.01)
    metadata = {k:v for k,v in vars(args).items() if k!="func"}
    metadata.update({"labels":LABELS, "descriptions":DESCRIPTIONS, "question":QUESTION,
                     "n_labels":n_labels,
                     "temperature":1., "status":"training", "test_evaluated":False,
                     "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
                     "interaction":args.interaction,
                     "implementation":"candidate-path LoRA; no shared evidence prefix in this engine"})
    write_json(out/"run_config.json",metadata)
    best = float("inf")
    history = []
    started = time.perf_counter()
    # Optional head-only initialization. Frozen pretrained weights remain frozen;
    # only the originally trainable adapter parameters are re-enabled afterwards.
    if args.head_warmup_steps:
        for parameter in body_params:
            parameter.requires_grad_(False)
        model.train()
        model.backbone.eval()
        for step in range(args.head_warmup_steps):
            ids = np.random.choice(len(rows), args.batch_size, replace=False)
            chunk = [rows[i] for i in ids]
            batch = tokenize_rows(chunk,tokenizer,args.mode,args.max_length)
            y = torch.tensor([r["label"] for r in chunk],device="cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits = model(batch,n_labels)
                loss = (torch.nn.functional.cross_entropy(logits.float(),y) if args.loss == "ce" else
                        ((logits.float().softmax(-1)-torch.nn.functional.one_hot(y,n_labels))**2).sum(-1).mean())
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params,1.)
            optim.step()
        for parameter in body_params:
            parameter.requires_grad_(True)
        print(f"Head-only warmup completed: {args.head_warmup_steps} steps",flush=True)
    total_steps = args.epochs * math.ceil(len(rows)/args.batch_size)
    warmup_steps = int(total_steps * args.warmup_ratio)
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(rows))
        loss_total = 0.
        for step,start in enumerate(range(0,len(order),args.batch_size)):
            if args.warmup_ratio > 0:
                scale = (min(1., (global_step+1)/max(1,warmup_steps)) if global_step < warmup_steps else
                         .5*(1+math.cos(math.pi*(global_step-warmup_steps)/max(1,total_steps-warmup_steps))))
                optim.param_groups[0]["lr"] = args.lr * scale
                optim.param_groups[1]["lr"] = head_lr * scale
            global_step += 1
            chunk = [rows[i] for i in order[start:start+args.batch_size]]
            # Consistent candidate permutation per batch; canonical IDs restored in evaluation.
            perm = (np.random.permutation(n_labels)
                    if args.mode=="dynamic" and args.candidate_randomization=="randomized"
                    else np.arange(n_labels))
            if chunk and "candidate_meanings" in chunk[0]:
                desc = [[f"Candidate ID: {i}\n{r['candidate_meanings'][i]}" for i in perm] for r in chunk]
            else:
                desc = [DESCRIPTIONS[i] for i in perm]
            batch = tokenize_rows(chunk,tokenizer,args.mode,args.max_length,desc)
            y = torch.tensor([int(np.flatnonzero(perm==r["label"])[0]) for r in chunk],device="cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=model(batch,n_labels)
                if args.loss in ("brier","ce_brier","ce_brier_08","ce_brier_09"):
                    brier=((logits.float().softmax(-1)-torch.nn.functional.one_hot(y,n_labels))**2).sum(-1).mean()
                    if args.loss=="brier": loss=brier
                    else:
                        w={"ce_brier":.5,"ce_brier_08":.2,"ce_brier_09":.1}[args.loss]
                        loss=(1-w)*torch.nn.functional.cross_entropy(logits.float(),y)+w*brier
                else:
                        loss=torch.nn.functional.cross_entropy(logits.float(),y)
                if args.interaction == "independent" and args.consistency_weight > 0:
                    perm2 = (np.random.permutation(n_labels)
                             if args.candidate_randomization=="randomized"
                             else np.arange(n_labels))
                    desc2 = [[f"Candidate ID: {i}\n{r['candidate_meanings'][i]}" for i in perm2] for r in chunk]
                    batch2 = tokenize_rows(chunk,tokenizer,args.mode,args.max_length,desc2)
                    logits2 = model(batch2,n_labels).float()
                    inv2 = np.argsort(perm2)
                    p1 = logits.float().softmax(-1)[:, np.argsort(perm)]
                    p2 = logits2.softmax(-1)[:, inv2]
                    loss = loss + args.consistency_weight * ((p1-p2)**2).sum(-1).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step()
            loss_total+=loss.item()*len(chunk)
            if step%100==0:
                print(f"{args.mode} epoch={epoch+1} step={step} loss={loss.item():.4f}",flush=True)
        logits=infer(model,tokenizer,selection,args)
        dev=metrics([r["label"] for r in selection],probabilities(logits))
        record={"epoch":epoch+1,"train_loss":loss_total/len(rows),"selection":dev,
                "elapsed_seconds":time.perf_counter()-started}
        history.append(record)
        write_json(out/"history.json",history)
        print(json.dumps({"epoch":epoch+1,"dev_accuracy":dev["accuracy"],"dev_nll":dev["nll"]}),flush=True)
        if dev["nll"]<best:
            best=dev["nll"]
            metadata["best_epoch"]=epoch+1
            save_model(model,tokenizer,out,metadata)
    # Restore selected adapter, never select using calibration/test.
    del optim,model,backbone,base
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    model,tokenizer,metadata=load_model(out)
    if args.dev_only:
        metadata["temperature"]=1.
        metadata["status"]="trained_development_only"
        metadata["calibration_status"]="not fitted; exploratory development run"
    else:
        cal=infer(model,tokenizer,calibration,args)
        metadata["temperature"]=fit_temperature(cal,[r["label"] for r in calibration])
        metadata["status"]="trained"
    metadata["training_seconds"]=time.perf_counter()-started
    write_json(out/"metadata.json",metadata)
    print("Finished development-only training; calibration and test not evaluated." if args.dev_only else
          "Finished training and independent calibration; test not evaluated.",flush=True)


def evaluate(args):
    out=Path(args.run)
    if (out/"test_results.json").exists():
        raise ValueError("Test already evaluated; preserve this run")
    model,tokenizer,meta=load_model(out)
    if meta["status"]!="trained":
        raise ValueError("Incomplete training")
    if audit(args.data)["splits"]!=json.loads((out/"data_audit.json").read_text())["splits"]:
        raise ValueError("Dataset fingerprint mismatch")
    rows=load_split(args.data,"test")
    y=np.array([r["label"] for r in rows])
    torch.cuda.synchronize()
    started=time.perf_counter()
    logits=infer(model,tokenizer,rows,args)
    torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    p=probabilities(logits,meta["temperature"])
    results={"raw":metrics(y,probabilities(logits)),"calibrated":metrics(y,p),
             "end_to_end_seconds":seconds,"batch_size":args.batch_size,
             "device":torch.cuda.get_device_name(0),"temperature":meta["temperature"],
             "mode":meta["mode"],"train_seconds":meta["training_seconds"]}
    if meta["mode"]=="dynamic" and meta.get("n_labels",3)==3:
        # Controls defined before test access; all rows retained in the primary result.
        check_rows=rows[:96]
        canon=infer(model,tokenizer,check_rows,args)
        swapped=infer(model,tokenizer,check_rows,args,[DESCRIPTIONS[i] for i in [2,0,1]])[:,[1,2,0]]
        alternate=infer(model,tokenizer,check_rows,args,
            ["The statement follows from the supplied clinical record.",
             "The statement conflicts with the supplied clinical record.",
             "The supplied clinical record leaves the statement undetermined."])
        results["robustness_96"]={"candidate_order_max_probability_diff":float(np.max(np.abs(probabilities(canon)-probabilities(swapped)))),
            "candidate_order_flip_rate":float(np.mean(canon.argmax(1)!=swapped.argmax(1))),
            "candidate_paraphrase_accuracy":float(np.mean(alternate.argmax(1)==y[:96])),
            "canonical_accuracy":float(np.mean(canon.argmax(1)==y[:96]))}
    np.savez_compressed(out/"test_predictions.npz",y=y,groups=np.array([r["group"] for r in rows]),probabilities=p,logits=logits)
    write_json(out/"test_results.json",results)
    meta["test_evaluated"]=True
    write_json(out/"metadata.json",meta)
    print(json.dumps(results,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    t=sub.add_parser("train")
    t.add_argument("--data",default="data")
    t.add_argument("--output",required=True)
    t.add_argument("--mode",choices=["dynamic","fixed"],default="dynamic")
    t.add_argument("--model",default=MODEL)
    t.add_argument("--revision",default=REVISION)
    t.add_argument("--epochs",type=int,default=3)
    t.add_argument("--batch-size",type=int,default=24)
    t.add_argument("--max-length",type=int,default=768)
    t.add_argument("--lora-rank",type=int,default=16)
    t.add_argument("--lr",type=float,default=2e-4)
    t.add_argument("--head-lr",type=float,default=None)
    t.add_argument("--head-warmup-steps",type=int,default=0)
    t.add_argument("--candidate-template",choices=["legacy","aligned_v2"],default="aligned_v2")
    t.add_argument("--warmup-ratio",type=float,default=0.)
    t.add_argument("--dev-only",action="store_true")
    t.add_argument("--loss",choices=["ce","brier","ce_brier","ce_brier_08","ce_brier_09"],default="ce")
    t.add_argument("--interaction",choices=["independent","pairwise_set","cross_attention","set_centered","set_attention"],default="independent")
    t.add_argument("--consistency-weight",type=float,default=0.0)
    t.add_argument("--candidate-randomization",choices=["randomized","canonical"],default="randomized")
    t.add_argument("--seed",type=int,default=17)
    t.set_defaults(func=train)
    e=sub.add_parser("evaluate")
    e.add_argument("--data",default="data")
    e.add_argument("--run",required=True)
    e.add_argument("--batch-size",type=int,default=48)
    e.add_argument("--max-length",type=int,default=768)
    e.set_defaults(func=evaluate)
    args=parser.parse_args()
    args.func(args)


if __name__=="__main__":
    main()
