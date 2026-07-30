from huggingface_hub import hf_hub_download
import os

ROOT_DIR = "/content/pmc_vqa"
os.makedirs(ROOT_DIR, exist_ok=True)

hf_hub_download("xmcmic/PMC-VQA", "train.csv", local_dir=ROOT_DIR, repo_type="dataset")
hf_hub_download("xmcmic/PMC-VQA", "train_2.csv", local_dir=ROOT_DIR, repo_type="dataset")
hf_hub_download("xmcmic/PMC-VQA", "images.zip", local_dir=ROOT_DIR, repo_type="dataset")
hf_hub_download("xmcmic/PMC-VQA", "images_2.zip", local_dir=ROOT_DIR, repo_type="dataset")

print("Download done")

#new cell
import pandas as pd
import os
import zipfile

# Load full training data
train1 = pd.read_csv("/content/pmc_vqa/train.csv")
train2 = pd.read_csv("/content/pmc_vqa/train_2.csv")
df = pd.concat([train1, train2], ignore_index=True)
df = df.dropna(subset=["Question", "Answer"])
df = df.drop_duplicates(subset=["Question", "Answer"])
df["Answer_clean"] = df["Answer"].str.replace(r'^[A-D]\s*:\s*', '', regex=True)

print(f"Total rows in CSVs: {len(df):,}")

# Check which images actually exist
available_images = set()
for zip_path in ["/content/pmc_vqa/images.zip", "/content/pmc_vqa/images_2.zip"]:
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            basename = os.path.basename(name)
            if basename:
                available_images.add(basename)

print(f"Total images in zips: {len(available_images):,}")

# Filter to rows with available images
df["has_image"] = df["Figure_path"].isin(available_images)
df_with_images = df[df["has_image"]].copy()

print(f"Rows with images: {len(df_with_images):,}")
print(f"Missing images: {len(df) - len(df_with_images):,}")

# Save
df_with_images.to_csv("/content/pmc_vqa_full_with_images.csv", index=False)

#new cell
import zipfile
import pandas as pd
import os
from tqdm import tqdm

ROOT = "/content/pmc_vqa"
OUT_DIR = f"{ROOT}/images"
os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv("/content/pmc_vqa_full_with_images.csv")
needed = set(df["Figure_path"].tolist())

ZIP_PATHS = [
    f"{ROOT}/images.zip",
    f"{ROOT}/images_2.zip",
]

already_done = set(os.listdir(OUT_DIR)) if os.path.exists(OUT_DIR) else set()
print(f"Already extracted: {len(already_done)}")

extracted = len(already_done)

for zip_path in ZIP_PATHS:
    if not os.path.exists(zip_path):
        print(f"⚠️  Skipping: {zip_path}")
        continue

    print(f"📦 Processing: {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        for name in tqdm(z.namelist()):
            base = os.path.basename(name)
            if not base:
                continue
            if base in needed and base not in already_done:
                with z.open(name) as src, open(os.path.join(OUT_DIR, base), "wb") as dst:
                    dst.write(src.read())
                already_done.add(base)
                extracted += 1

print(f"\n Total extracted : {extracted:,} / {len(needed):,} needed")
print(f" Still missing   : {len(needed) - extracted:,}")
