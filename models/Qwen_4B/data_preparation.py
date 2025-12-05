#!/usr/bin/env python3
"""
Data preparation utility for SFT/DPO training.
Ensures all required data splits and DPO pairs are generated and stored in data_cache/.

Usage:
    python data_preparation.py --include_similarity_report [True|False] --type [classification|comparison]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Configurable paths
CATEGORIES = ["CS_CV", "CS_CL", "CS_LG", "CS_RO", "CS_CR", "CS_AI"]
CATEGORY_ROOT = Path("data_cache/categories")
SIMILARITY_ROOT = Path("data_cache/similiarity_aware_categories")

GENERATE_ALL_CATEGORIES = "embedding/similiarity_report/generate_all_categories.py"
GENERATE_DPO_PAIRS = "embedding/similiarity_report/generate_dpo_pairs.py"
GENERATE_SIMILARITY_REPORTS = "embedding/similiarity_report/generate_similarity_reports.py"


def run_python(script, args=None):
    cmd = [sys.executable, script]
    if args:
        cmd += args
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def ensure_categories():
    print(f"[CHECK] Checking if {CATEGORY_ROOT} exists...")
    if not CATEGORY_ROOT.exists():
        print(f"[INFO] {CATEGORY_ROOT} not found. Generating base category splits...")
        run_python(GENERATE_ALL_CATEGORIES)
    else:
        print(f"[OK] {CATEGORY_ROOT} already present.")


def ensure_similarity_aware():
    print(f"[CHECK] Checking if {SIMILARITY_ROOT} exists...")
    if not SIMILARITY_ROOT.exists():
        print(f"[INFO] {SIMILARITY_ROOT} not found. Generating similarity-aware splits...")
        run_python(GENERATE_SIMILARITY_REPORTS)
    else:
        print(f"[OK] {SIMILARITY_ROOT} already present.")


def ensure_dpo_pairs(root, categories, dpo_type, include_similarity_report):
    for cat in categories:
        dpo_dir = root / cat / "dpo" / dpo_type
        print(f"[CHECK] Checking if {dpo_dir} exists...")
        if not dpo_dir.exists():
            print(f"[INFO] {dpo_dir} not found. Generating DPO pairs for {cat} ({dpo_type})...")
            args = [
                "--task", dpo_type,
                "--category-root", str(root),
                "--categories", cat,
                "--overwrite"
            ]
            if include_similarity_report:
                args.append("--include-similarity-report")
            run_python(GENERATE_DPO_PAIRS, args)
        else:
            print(f"[OK] {dpo_dir} already present.")



def prepare_data(args):
    """
    args: Namespace with .data_variant (str: base/sim) and .type (str: classification/comparison)
    """
    print("[STEP 1] Ensuring base category splits...")
    ensure_categories()

    if args.data_variant == "base":
        print("[STEP 2] Ensuring DPO pairs for base splits...")
        ensure_dpo_pairs(CATEGORY_ROOT, CATEGORIES, args.type, include_similarity_report=False)
    elif args.data_variant == "sim":
        print("[STEP 2] Ensuring similarity-aware splits...")
        ensure_similarity_aware()
        print("[STEP 3] Ensuring DPO pairs for similarity-aware splits...")
        ensure_dpo_pairs(SIMILARITY_ROOT, CATEGORIES, args.type, include_similarity_report=True)
    else:
        raise ValueError(f"Unknown data_variant: {args.data_variant}")

    print("[DONE] Data preparation complete.")


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT/DPO data for training.")
    parser.add_argument("--data_variant", choices=["base", "sim"], required=True, help="Data variant: base or sim (similarity-aware splits).")
    parser.add_argument("--type", choices=["classification", "comparison"], required=True, help="DPO pair type.")
    args = parser.parse_args()
    prepare_data(args)


if __name__ == "__main__":
    main()
