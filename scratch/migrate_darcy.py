import json
import os
import pandas as pd

RESULTS_JSON = "results.json"
RESULTS_TSV = "results.tsv"

def migrate_json():
    if not os.path.exists(RESULTS_JSON):
        print(f"File {RESULTS_JSON} not found.")
        return

    with open(RESULTS_JSON, "r") as f:
        data = json.load(f)

    # 1. Remove flawed darcy_2d
    # 2. Rename darcy_2d_fix to darcy_2d
    new_data = []
    removed_count = 0
    renamed_count = 0

    for entry in data:
        bm = entry.get("benchmark")
        if bm == "darcy_2d":
            # These are the flawed ones. We remove them.
            removed_count += 1
            continue
        elif bm == "darcy_2d_fix":
            entry["benchmark"] = "darcy_2d"
            renamed_count += 1
            new_data.append(entry)
        else:
            new_data.append(entry)

    with open(RESULTS_JSON, "w") as f:
        json.dump(new_data, f, indent=2)
    
    print(f"JSON: Removed {removed_count} flawed entries, renamed {renamed_count} entries.")

def migrate_tsv():
    if not os.path.exists(RESULTS_TSV):
        print(f"File {RESULTS_TSV} not found.")
        return

    try:
        df = pd.read_csv(RESULTS_TSV, sep="\t")
        
        # 1. Remove flawed darcy_2d
        # Flawed ones typically have high error, but the user said "all entries for the original flawed darcy_2d"
        # However, if some rows are darcy_2d and some are darcy_2d_fix, we can distinguish.
        
        # Actually, if I just do row by row:
        removed_mask = df["benchmark"] == "darcy_2d"
        removed_count = removed_mask.sum()
        df = df[~removed_mask].copy()
        
        renamed_mask = df["benchmark"] == "darcy_2d_fix"
        renamed_count = renamed_mask.sum()
        df.loc[renamed_mask, "benchmark"] = "darcy_2d"
        
        df.to_csv(RESULTS_TSV, sep="\t", index=False)
        print(f"TSV: Removed {removed_count} flawed rows, renamed {renamed_count} rows.")
    except Exception as e:
        print(f"TSV Migration failed: {e}")

if __name__ == "__main__":
    migrate_json()
    migrate_tsv()
