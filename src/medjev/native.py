"""SemIf-inspired direct label logits with an actual shared-prefix cache path."""
from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .data import DESCRIPTIONS, QUESTION, load_split, dev_partition, write_json
from .metrics import metrics, probabilities, fit_temperature
from .gpu import MODEL, REVISION


class NativeEngine:
    def __init__(self, model_id=MODEL, revision=REVISION, device="cuda", dtype=None):
        self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision)
        self.tokenizer.padding_side="right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,
            torch_dtype=dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32),
            attn_implementation="sdpa").to(device).eval()
        self.slots=[]
        for letter in ["A","B","C"]:
            ids=self.tokenizer.encode(letter,add_special_tokens=False)
            if len(ids)!=1:
                raise ValueError("Answer label must be one token")
            self.slots.append(ids[0])

    def prompt(self,row):
        options="\n".join(f"{letter}: {d}" for letter,d in zip("ABC",DESCRIPTIONS))
        content=f"Clinical evidence:\n{row['premise']}\n\nStatement:\n{row['hypothesis']}\n\n{QUESTION}\n{options}\nRespond with only A, B, or C."
        return self.tokenizer.apply_chat_template(
            [{"role":"system","content":"Classify the statement using only the provided clinical evidence."},
             {"role":"user","content":content}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)+"Answer:"

    @torch.inference_mode()
    def fresh(self,rows,batch_size=32):
        results=[]
        for start in range(0,len(rows),batch_size):
            batch=self.tokenizer([self.prompt(r) for r in rows[start:start+batch_size]],
                                 padding=True,return_tensors="pt").to(self.device)
            hidden=self.model.model(**batch,use_cache=False).last_hidden_state
            last=batch.attention_mask.sum(1)-1
            states=hidden[torch.arange(hidden.shape[0],device=self.device),last]
            scores=self.model.lm_head(states).float()[:,self.slots]
            results.append(scores.cpu().numpy())
        return np.concatenate(results)

    @torch.inference_mode()
    def shared(self,rows):
        groups=defaultdict(list)
        for i,r in enumerate(rows):
            groups[r["premise"]].append((i,r))
        result=np.zeros((len(rows),3))
        stats={"groups":len(groups),"prefix_tokens":0,"suffix_tokens":0,"full_tokens_without_cache":0}
        for pairs in groups.values():
            ids=[self.tokenizer.encode(self.prompt(r),add_special_tokens=True) for _,r in pairs]
            # Exact token common prefix; leave at least one suffix token per query.
            n=0
            for token_set in zip(*ids):
                if len(set(token_set))!=1:
                    break
                n+=1
            n=min(n,min(map(len,ids))-1)
            if n<1:
                values=self.fresh([r for _,r in pairs])
            else:
                prefix=torch.tensor([ids[0][:n]],device=self.device)
                pref=self.model.model(input_ids=prefix,use_cache=True)
                cache=pref.past_key_values
                cache.batch_repeat_interleave(len(ids))
                suffixes=[x[n:] for x in ids]
                maxlen=max(map(len,suffixes))
                tokens=torch.full((len(ids),maxlen),self.tokenizer.pad_token_id,device=self.device,dtype=torch.long)
                mask=torch.zeros((len(ids),n+maxlen),device=self.device,dtype=torch.long)
                positions=torch.zeros_like(tokens)
                for j,s in enumerate(suffixes):
                    tokens[j,:len(s)]=torch.tensor(s,device=self.device)
                    mask[j,:n+len(s)]=1
                    positions[j,:len(s)]=torch.arange(n,n+len(s),device=self.device)
                hidden=self.model.model(input_ids=tokens,attention_mask=mask,position_ids=positions,
                                        past_key_values=cache,use_cache=True).last_hidden_state
                states=hidden[torch.arange(len(ids),device=self.device),
                              torch.tensor([len(s)-1 for s in suffixes],device=self.device)]
                values=self.model.lm_head(states).float()[:,self.slots].cpu().numpy()
                stats["prefix_tokens"]+=n
                stats["suffix_tokens"]+=sum(map(len,suffixes))
                stats["full_tokens_without_cache"]+=sum(map(len,ids))
            for (i,_),v in zip(pairs,values):
                result[i]=v
        return result,stats

    @torch.inference_mode()
    def generate(self,system,user,max_new_tokens=180):
        text=self.tokenizer.apply_chat_template([{"role":"system","content":system},{"role":"user","content":user}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        batch=self.tokenizer(text,return_tensors="pt").to(self.device)
        generated=self.model.generate(**batch,do_sample=False,max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(generated[0,batch.input_ids.shape[1]:],skip_special_tokens=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data")
    p.add_argument("--output",required=True)
    p.add_argument("--batch-size",type=int,default=32)
    args=p.parse_args()
    out=Path(args.output)
    if (out/"test_results.json").exists():
        raise ValueError("Already evaluated; choose a new run only for justified correction")
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    engine=NativeEngine()
    _,cal=dev_partition(load_split(args.data,"dev"))
    cal_logits=engine.fresh(cal,args.batch_size)
    temp=fit_temperature(cal_logits,[r["label"] for r in cal])
    test=load_split(args.data,"test")
    engine.fresh(test[:3],3)
    torch.cuda.synchronize()
    start=time.perf_counter()
    logits=engine.fresh(test,args.batch_size)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    # Same natural 96-query cohort; no fabricated repetitions for cache benchmark.
    subset=test[:96]
    torch.cuda.synchronize()
    start=time.perf_counter()
    serial=engine.fresh(subset,1)
    torch.cuda.synchronize()
    serial_seconds=time.perf_counter()-start
    start=time.perf_counter()
    shared,stats=engine.shared(subset)
    torch.cuda.synchronize()
    shared_seconds=time.perf_counter()-start
    y=np.array([r["label"] for r in test])
    prob=probabilities(logits,temp)
    result={"model":MODEL,"revision":REVISION,"temperature":temp,
        "raw":metrics(y,probabilities(logits)),"calibrated":metrics(y,prob),
        "fresh_batched_seconds":elapsed,"batch_size":args.batch_size,
        "cache_96":{"serial_fresh_seconds":serial_seconds,"shared_seconds":shared_seconds,
            "max_probability_difference":float(np.max(np.abs(probabilities(serial)-probabilities(shared)))),
            "argmax_flip_rate":float(np.mean(serial.argmax(1)!=shared.argmax(1))),"stats":stats},
        "note":"Zero generated output tokens for decision readout; shared cache benchmark is a separate natural 96-query cohort."}
    write_json(out/"test_results.json",result)
    np.savez_compressed(out/"test_predictions.npz",y=y,groups=np.array([r["group"] for r in test]),probabilities=prob,logits=logits)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
