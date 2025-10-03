import argparse

import yaml

from rla.models.tiny_tcn import TinyTCN

# from rla.losses.huber import huber_loss  # keep commented until used


def main(cfg: dict) -> None:
    # NOTE: we'll wire training later; for now keep the stub clean for Ruff/mypy
    d_in = int(cfg["data"]["d_in"])
    d_out = int(cfg["data"]["d_out"])
    channels = int(cfg["model"]["tcn"]["channels"])
    model = TinyTCN(d_in=d_in, d_out=d_out, channels=channels)
    _param_count = sum(p.numel() for p in model.parameters())
    print(f"Training stub OK. Params: {_param_count}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # inject minimal data dims for now:
    cfg.setdefault("data", {})
    cfg["data"].setdefault("d_in", 1)
    cfg["data"].setdefault("d_out", 1)

    main(cfg)
