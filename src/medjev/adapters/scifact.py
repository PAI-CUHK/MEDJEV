import json
from pathlib import Path
from .common import DecisionRecord

_C=[{"id":"support","meaning":"the cited evidence supports the claim"},{"id":"contradict","meaning":"the cited evidence contradicts the claim"},{"id":"neutral","meaning":"the cited evidence does not determine the claim"}]

def load_scifact(root, split="train"):
    root=Path(root); p=root/"data"/f"claims_{split}.jsonl"; out=[]
    if not p.exists(): return out
    corpus={x["doc_id"]:x for x in (json.loads(source_line) for source_line in (root/"data"/"corpus.jsonl").read_text(encoding="utf-8").splitlines())}
    for line in p.read_text(encoding="utf-8").splitlines():
        x=json.loads(line)
        # SciFact claims files do not contain gold labels for train/dev/test.
        # Do not silently turn missing labels into neutral; labels require the
        # repository's evidence/annotation files and are resolved by the audit.
        ev=x.get("evidence",{}); labels=[e.get("label","").lower() for vals in ev.values() for e in vals]
        label=labels[0] if labels and all(z==labels[0] for z in labels) else ("neutral" if not labels and split != "test" else "")
        spans=[]
        for did, vals in ev.items():
            doc=corpus.get(int(did), corpus.get(did,{})); abst=doc.get("abstract",[])
            for e in vals:
                spans.extend(str(abst[i]) for i in e.get("sentences",[]) if i < len(abst))
        state="\n".join(str(corpus.get(int(d),corpus.get(d,{})).get("title",""))+"\n"+"\n".join(corpus.get(int(d),corpus.get(d,{})).get("abstract",[])) for d in x.get("cited_doc_ids",[]) if corpus.get(int(d),corpus.get(d)))
        out.append(DecisionRecord(f"scifact:{x['id']}","scifact",state,x.get("claim",""),_C,label,spans,[],str(x.get("cited_doc_ids",x['id'])),split,label,{"cited_doc_ids":x.get("cited_doc_ids",[]),"evidence":ev,"label_status":"inferred_from_evidence" if labels else ("neutral_without_evidence" if split != "test" else "unlabeled_test")}))
    return out
