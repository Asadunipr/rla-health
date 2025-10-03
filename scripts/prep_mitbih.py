import argparse
from pathlib import Path

import wfdb


def main(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Example: download a small subset for smoke test; later, full 48 records
    records = wfdb.get_record_list("mitdb")
    # TODO: fetch signals, annotations; save as npz windows (train normals, val normals, test all)
    print(f"Downloaded list of {len(records)} records. Implement windowing next.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.out)
