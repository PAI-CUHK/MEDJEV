import argparse
from .data import audit, write_json


def main():
    p = argparse.ArgumentParser(description="MedJEV local research runner")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--data", default="dataset")
    a.add_argument("--output", default="artifacts/data_audit.json")
    a.set_defaults(func=lambda args: write_json(args.output, audit(args.data)))
    t = sub.add_parser("train", help="Frozen-encoder shared decision head + controls")
    t.add_argument("--data", default="dataset")
    t.add_argument("--output", required=True)
    t.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    t.add_argument("--revision", default=None)
    t.add_argument("--device", default="cpu")
    t.add_argument("--cache", default="cache/features")
    t.add_argument("--max-length", type=int, default=256)
    t.add_argument("--encode-batch", type=int, default=32)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--rank", type=int, default=128)
    t.add_argument("--lr", type=float, default=.001)
    t.add_argument("--seed", type=int, default=17)
    t.add_argument("--loss", choices=["ce", "brier", "ce_brier", "ce_brier_08", "ce_brier_09"], default="ce")
    def train(args):
        from .experiment import train_experiment
        train_experiment(args)
    t.set_defaults(func=train)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", required=True)
    e.add_argument("--data", default="dataset")
    e.add_argument("--device", default="cpu")
    e.add_argument("--overwrite", action="store_true")
    def evaluate(args):
        from .experiment import evaluate_experiment
        evaluate_experiment(args)
    e.set_defaults(func=evaluate)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
