from __future__ import annotations
import json
import time
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from .data import LABELS, DESCRIPTIONS, QUESTION, digest, load_split, dev_partition, audit, write_json
from .metrics import metrics, probabilities, fit_temperature, paired_bootstrap
from .model import TextEncoder, DecisionHead, PooledClassifier, pad_evidence


def candidate_text(hypothesis, description, question=QUESTION):
    return f"{question}\nStatement: {hypothesis}\nAnswer meaning: {description}"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_cache(rows, encoder, directory, batch_size=32, descriptions=None, question=QUESTION):
    descriptions = descriptions or DESCRIPTIONS
    fingerprint = digest(json.dumps({"ids": [r["id"] for r in rows],
        "content": [digest(r["premise"] + "\0" + r["hypothesis"]) for r in rows],
        "encoder": encoder.model_id, "revision": encoder.revision,
        "max_length": encoder.max_length, "descriptions": descriptions, "question": question}, sort_keys=True))
    path = Path(directory) / (fingerprint + ".pt")
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    premises = list(dict.fromkeys(r["premise"] for r in rows))
    lookup = {s: i for i, s in enumerate(premises)}
    print(f"Encoding {len(premises)} unique premises and {len(rows)} queries...", flush=True)
    started = time.perf_counter()
    truncated_before = encoder.truncated
    p = encoder.encode(premises, batch_size, tokens=True)
    h = encoder.encode([r["hypothesis"] for r in rows], batch_size)
    c = encoder.encode([candidate_text(r["hypothesis"], d, question) for r in rows for d in descriptions], batch_size)
    cache = {"premises": p, "index": [lookup[r["premise"]] for r in rows], "hypotheses": h,
             "candidates": c.reshape(len(rows), len(descriptions), -1),
             "labels": torch.tensor([r["label"] for r in rows]),
             "metadata": {"fingerprint": fingerprint, "encoder_seconds": time.perf_counter()-started,
                          "truncated_sequences": encoder.truncated-truncated_before,
                          "encoding_includes_baseline_hypotheses": True}}
    torch.save(cache, path)
    return cache


def make_batch(cache, indices, device, dynamic):
    evidence, mask = pad_evidence([cache["premises"][cache["index"][i]] for i in indices], device)
    field = "candidates" if dynamic else "hypotheses"
    return evidence, mask, cache[field][indices].float().to(device)


@torch.inference_mode()
def predict_head(model, cache, device, dynamic, batch_size=128):
    model.eval()
    logits = []
    for start in range(0, len(cache["labels"]), batch_size):
        ids = list(range(start, min(start + batch_size, len(cache["labels"]))))
        logits.append(model(*make_batch(cache, ids, device, dynamic)).cpu().numpy())
    return np.concatenate(logits)


def fit_head(train, dev, config, dynamic=True):
    seed_all(config["seed"])
    width = train["hypotheses"].shape[-1]
    cls = DecisionHead if dynamic else PooledClassifier
    model = cls(width, config["rank"]).to(config["device"])
    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=.01)
    best, best_state, history = float("inf"), None, []
    started = time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        order = np.random.permutation(len(train["labels"]))
        total = 0.
        for start in range(0, len(order), config["batch_size"]):
            ids = order[start:start+config["batch_size"]].tolist()
            logits = model(*make_batch(train, ids, config["device"], dynamic))
            y = train["labels"][ids].to(config["device"])
            if config["loss"] in ("brier", "ce_brier", "ce_brier_08", "ce_brier_09"):
                target = torch.nn.functional.one_hot(y, 3)
                brier = ((logits.softmax(-1) - target)**2).sum(-1).mean()
                if config["loss"] == "brier": loss = brier
                else:
                    w = {"ce_brier": .5, "ce_brier_08": .2, "ce_brier_09": .1}[config["loss"]]
                    loss = (1-w) * torch.nn.functional.cross_entropy(logits, y) + w * brier
            else:
                loss = torch.nn.functional.cross_entropy(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            total += loss.item() * len(ids)
        result = metrics(dev["labels"].numpy(), probabilities(predict_head(model, dev, config["device"], dynamic)))
        history.append({"epoch": epoch+1, "train_loss": total / len(order), "dev": result})
        print(f"{'dynamic' if dynamic else 'fixed'} epoch {epoch+1}: dev_acc={result['accuracy']:.4f} dev_nll={result['nll']:.4f}", flush=True)
        if result["nll"] < best:
            best = result["nll"]
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history, time.perf_counter()-started


def lexical_text(rows, hypothesis_only=False):
    return [r["hypothesis"] if hypothesis_only else "premise: " + r["premise"] + " hypothesis: " + r["hypothesis"] for r in rows]


def train_experiment(args):
    out = Path(args.output)
    if (out / "manifest.json").exists():
        raise ValueError("Run directory already contains a manifest; choose a fresh output to preserve audit trail")
    out.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit(args.data)
    write_json(out / "data_audit.json", dataset_audit)
    train = load_split(args.data, "train")
    selection, calibration = dev_partition(load_split(args.data, "dev"))
    config = vars(args).copy()
    config.pop("func", None)
    config["labels"] = LABELS
    config["descriptions"] = DESCRIPTIONS
    config["question"] = QUESTION
    config["status"] = "training"
    config["test_evaluated"] = False
    write_json(out / "manifest.json", config)
    print("Training lexical and hypothesis-only controls...", flush=True)
    results = {}
    for name, hyp_only in [("tfidf_pair", False), ("tfidf_hypothesis_only", True)]:
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=60000)
        x = vec.fit_transform(lexical_text(train, hyp_only))
        clf = LogisticRegression(C=1., max_iter=500)
        clf.fit(x, [r["label"] for r in train])
        cal_logits = clf.decision_function(vec.transform(lexical_text(calibration, hyp_only)))
        temp = fit_temperature(cal_logits, [r["label"] for r in calibration])
        joblib.dump({"vectorizer": vec, "classifier": clf, "temperature": temp, "hypothesis_only": hyp_only}, out/(name+".joblib"))
        val_logits = clf.decision_function(vec.transform(lexical_text(selection, hyp_only)))
        results[name] = {"selection": metrics([r["label"] for r in selection], probabilities(val_logits)),
                         "temperature": temp}
    encoder = TextEncoder(args.encoder, args.revision, args.device, args.max_length, args.threads)
    config["resolved_encoder_revision"] = getattr(encoder.model.config, "_commit_hash", None)
    write_json(out/"manifest.json", config)
    cache_args = (encoder, args.cache, args.encode_batch)
    tr, dv, ca = [feature_cache(rows, *cache_args) for rows in [train, selection, calibration]]
    config["feature_metadata"] = {"train": tr["metadata"], "selection": dv["metadata"], "calibration": ca["metadata"]}
    for name, dynamic in [("pooled_classifier", False), ("medjev_shared", True)]:
        model, history, seconds = fit_head(tr, dv, config, dynamic)
        logits = predict_head(model, ca, args.device, dynamic)
        temp = fit_temperature(logits, ca["labels"].numpy())
        torch.save({"state_dict": {k:v.cpu() for k,v in model.state_dict().items()},
                    "width": encoder.width, "rank": args.rank, "temperature": temp, "dynamic": dynamic}, out/(name+".pt"))
        results[name] = {"selection": min(history, key=lambda r:r["dev"]["nll"])["dev"],
                         "temperature": temp, "training_seconds": seconds, "history": history}
        write_json(out/"development_results.json", results)
    config["status"] = "trained"
    write_json(out/"manifest.json", config)
    print("Training complete. Test is sealed. Run evaluate explicitly for the final comparison.", flush=True)


def evaluate_experiment(args):
    out = Path(args.run)
    if (out / "test_results.json").exists() and not args.overwrite:
        raise ValueError("Test already evaluated. Use --overwrite only for a documented bug fix.")
    cfg = json.loads((out/"manifest.json").read_text())
    if cfg["status"] != "trained":
        raise ValueError("Training did not finish")
    current_audit = audit(args.data)
    saved_audit = json.loads((out/"data_audit.json").read_text())
    if current_audit["splits"] != saved_audit["splits"]:
        raise ValueError("Dataset changed since training")
    rows = load_split(args.data, "test")
    y = np.array([r["label"] for r in rows])
    groups = [r["group"] for r in rows]
    results, all_probs = {}, {}
    for name in ["tfidf_pair", "tfidf_hypothesis_only"]:
        obj = joblib.load(out/(name+".joblib"))
        logits = obj["classifier"].decision_function(obj["vectorizer"].transform(lexical_text(rows, obj["hypothesis_only"])))
        p = probabilities(logits, obj["temperature"])
        all_probs[name] = p
        results[name] = {"raw": metrics(y, probabilities(logits)), "calibrated": metrics(y, p)}
    encoder = TextEncoder(cfg["encoder"], cfg.get("revision"), args.device, cfg["max_length"], cfg["threads"])
    cache = feature_cache(rows, encoder, cfg["cache"], cfg["encode_batch"])
    for name in ["pooled_classifier", "medjev_shared"]:
        ck = torch.load(out/(name+".pt"), map_location="cpu", weights_only=True)
        model = (DecisionHead if ck["dynamic"] else PooledClassifier)(ck["width"], ck["rank"]).to(args.device)
        model.load_state_dict(ck["state_dict"])
        started = time.perf_counter()
        logits = predict_head(model, cache, args.device, ck["dynamic"])
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        head_seconds = time.perf_counter()-started
        p = probabilities(logits, ck["temperature"])
        all_probs[name] = p
        results[name] = {"raw": metrics(y, probabilities(logits)), "calibrated": metrics(y, p),
                         "head_only_seconds": head_seconds,
                         "timing_warning": "Feature-cache readout timing, not end-to-end generation speedup."}
    intervals = {name: paired_bootstrap(y, all_probs["medjev_shared"], p, groups)
                 for name,p in all_probs.items() if name != "medjev_shared"}
    write_json(out/"test_results.json", {"methods": results, "paired_intervals": intervals,
        "test_feature_metadata": cache["metadata"],
        "limitations": ["Single seed", "Frozen general-domain encoder unless manifest says otherwise",
                       "Only three MedNLI relations trained; not general clinical question understanding",
                       "No LLM closed-loop effectiveness measured by this experiment",
                       "Exact premise grouping cannot establish patient disjointness"]})
    np.savez_compressed(out/"test_predictions.npz", y=y, groups=np.array(groups), **all_probs)
    cfg["test_evaluated"] = True
    write_json(out/"manifest.json", cfg)
    print(json.dumps({k:v["calibrated"] for k,v in results.items()}, indent=2), flush=True)
    make_report(out)


def make_report(out):
    out = Path(out)
    result = json.loads((out/"test_results.json").read_text())
    cfg = json.loads((out/"manifest.json").read_text())
    lines = ["# MedJEV 实测结果", "", "这是本次运行结果，不是整个规划已完成的证明。", "",
             f"- 编码器：{cfg['encoder']}", f"- 训练种子：{cfg['seed']}",
             "- 测试：官方 MedNLI test；开发集按完全相同前提分组，独立划分模型选择与校准。",
             "", "| 方法 | Accuracy | Macro-F1 | NLL | Brier | ECE |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, item in result["methods"].items():
        m = item["calibrated"]
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['nll']:.4f} | {m['brier']:.4f} | {m['ece_10']:.4f} |")
    lines += ["", "## 与对照的配对准确率差", ""]
    for name, interval in result["paired_intervals"].items():
        lo, hi = interval["ci95"]
        lines.append(f"- MedJEV − {name}: {interval['point']:.4f}, 95% 前提分组 bootstrap 区间 [{lo:.4f}, {hi:.4f}]。")
    interval = result["paired_intervals"]["pooled_classifier"]
    if interval["ci95"][0] > 0:
        verdict = "本次 MedNLI 实验支持共享动态判断头比固定池化分类头更准确；尚不能外推到未见任务、临床结果或 LLM 回答改善。"
    else:
        verdict = "本次实验没有建立共享动态判断头优于固定分类头的证据。接口可运行不等于方法优越，需保留这一结果。"
    lines += ["", "## 结论", "", verdict, "", "## 局限", "",
              "只评估三类临床文本推断；未验证真实患病风险、开放式诊断、LLM 修订效果或相对生成式 LLM 的端到端加速。",
              "缓存后的判断头计时不能当作整个模型推理时延。单种子结果需要后续复验。"]
    (out/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
