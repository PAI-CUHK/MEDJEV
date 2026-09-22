import json
from pathlib import Path
from .common import DecisionRecord

def load_medqa(path, split="train"):
    p=Path(path); out=[]
    if p.is_dir(): p=p/f"{split}.jsonl"
    if not p.exists(): return out
    for i,line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if not line.strip(): continue
        x=json.loads(line); opts=x.get("options",x.get("choices",[])); answer=x.get("answer_idx",x.get("label",x.get("answer","")))
        if isinstance(opts,dict): opts=[opts[k] for k in sorted(opts)]
        c=[{"id":str(j),"meaning":str(v)} for j,v in enumerate(opts)]
        target=str(answer)
        if isinstance(answer,int): target=str(answer)
        elif isinstance(answer,str) and len(answer)==1 and answer.upper() in "ABCDE": target=str("ABCDE".index(answer.upper()))
        out.append(DecisionRecord(f"medqa:{split}:{i}","medqa",str(x.get("question",x.get("question_text",""))),str(x.get("question",x.get("question_text",""))),c,target,source_group=f"medqa:{split}:{i}",split=split,original_label=target,metadata=x))
    return out
