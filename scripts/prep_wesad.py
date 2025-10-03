import argparse
from pathlib import Path


def main(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # TODO: unzip WESAD files into data/wesad/raw, resample to 8 Hz, sync channels,
    # split LOSO folds, window into (L, d) arrays, save npz per fold
    print("WESAD prep stub OK.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    main(args.out)
