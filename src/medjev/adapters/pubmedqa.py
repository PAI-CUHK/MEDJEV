import json
from pathlib import Path
from .common import DecisionRecord

_C = [{"id":"yes","meaning":"the evidence supports a yes answer"},{"id":"no","meaning":"the evidence supports a no answer"},{"id":"maybe","meaning":"the evidence is insufficient or uncertain"}]

def load_pubmedqa(root, split="train"):
    root=Path(root); name={"train":"pqaa_train_set.json","dev":"pqaa_dev_set.json","test":"test_set.json"}[split]
    data=json.loads((root/"data"/name).read_text(encoding="utf-8"))
    out=[]
    for pmid, item in data.items():
        label=str(item.get("final decision", item.get("final_decision", item.get("LABEL", "")))).lower()
        if label not in {"yes","no","maybe"}: continue
        contexts=item.get("CONTEXTS", item.get("contexts", [])); state="\n".join(contexts) if isinstance(contexts,list) else str(contexts)
        q=item.get("QUESTION", item.get("question", ""))
        out.append(DecisionRecord(f"pubmedqa:{pmid}","pubmedqa",state,q,_C,label,source_group=str(pmid),split=split,original_label=label,metadata={"pmid":pmid}))
    return out
