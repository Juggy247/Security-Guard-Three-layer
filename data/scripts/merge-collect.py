import pandas as pd

from pathlib import Path

DATA = Path(__file__).resolve().parent.parent   # the data/ folder
RAW, INTERIM, FINAL = DATA / "raw", DATA / "interim", DATA / "final"

# load all new sources
deepset_leaking = pd.read_csv(RAW / "deepset_leaking.csv")
deepset_jailbreak = pd.read_csv(RAW / "deepset_jailbreak_only.csv")
deepset_safe = pd.read_csv(RAW / "deepset_safe.csv")

# clean up the extra text_clean column if present
for df in [deepset_leaking, deepset_jailbreak, deepset_safe]:
    if "text_clean" in df.columns:
        df.drop(columns=["text_clean"], inplace=True)

# load your existing full pieces (the ones you already have)
existing_partial = pd.read_csv(INTERIM / "dataset_partial.csv")
existing_encoded = pd.read_csv(RAW / "encoded_attacks.csv")
existing_leaking = pd.read_csv(RAW / "leaking_attacks.csv")
existing_jailbreak = pd.read_csv(RAW / "manual_jailbreaks.csv")
existing_tricky_safe = pd.read_csv(RAW / "tricky_safe.csv")

# combine everything
all_data = pd.concat([
    existing_partial,
    existing_encoded,
    existing_leaking,
    existing_jailbreak,
    existing_tricky_safe,
    deepset_leaking,
    deepset_jailbreak,
    deepset_safe,
], ignore_index=True)

# dedup just in case
all_data = all_data.drop_duplicates(subset=["prompt"]).reset_index(drop=True)

print(f"Total combined: {len(all_data)}")
print(all_data["attack_type"].value_counts())
print(f"\nLabel balance:\n{all_data['label'].value_counts()}")

all_data.to_csv(INTERIM / "combined_dataset_v2.csv", index=False)
print("\nSaved combined_dataset_v2.csv")