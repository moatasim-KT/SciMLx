import json
import os
import pandas as pd

RESULTS_JSON = "results.json"
RESULTS_TSV = "results.tsv"

def migrate_json():
    print(f"Migrating {RESULTS_JSON}...")
    with open(RESULTS_JSON, "r") as f:
        data = json.load(f)

    original_count = len(data)
    new_data = []
    renamed_count = 0
    deleted_count = 0

    for entry in data:
        bm = entry.get("benchmark")
        
        # 1. Delete legacy broken NS entries
        if bm == "ns_2d":
            deleted_count += 1
            continue
            
        # 2. Rename ns_2d to ns_2d
        elif bm == "ns_2d":
            entry["benchmark"] = "ns_2d"
            # Also update name and ID if they contain the old benchmark name prefix
            # Actually, just the benchmark field is enough for Dashboard routing.
            renamed_count += 1
        
        # Also check conclusion/config nested fields
        if entry.get("config", {}).get("benchmark") == "ns_2d":
            entry["config"]["benchmark"] = "ns_2d"
        
        new_data.append(entry)

    with open(RESULTS_JSON, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"  Deleted {deleted_count} broken 'ns_2d' legacy entries.")
    print(f"  Renamed {renamed_count} 'ns_2d' entries to 'ns_2d'.")
    print(f"  Total entries: {original_count} -> {len(new_data)}")

def migrate_tsv():
    print(f"Migrating {RESULTS_TSV}...")
    if not os.path.exists(RESULTS_TSV):
        print("  TSV not found, skipping.")
        return

    df = pd.read_csv(RESULTS_TSV, sep="\t")
    original_count = len(df)

    # Delete broken
    df = df[df["benchmark"] != "ns_2d"]
    deleted_count = original_count - len(df)

    # Rename fix
    df.loc[df["benchmark"] == "ns_2d", "benchmark"] = "ns_2d"
    renamed_count = (df["benchmark"] == "ns_2d").sum() # This isn't quite right for count, but let's just do it.

    df.to_csv(RESULTS_TSV, sep="\t", index=False)
    print(f"  Deleted {deleted_count} legacy rows.")
    print(f"  Processed {len(df)} remaining rows.")

if __name__ == "__main__":
    migrate_json()
    migrate_tsv()
