import os
import json
import requests
import zipfile
from datasets import load_dataset

os.makedirs("/content/slake/imgs", exist_ok=True)

# Download SLAKE images from official source
print("📥 Downloading SLAKE images...")
url = "https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip"
zip_path = "/content/slake/imgs.zip"

response = requests.get(url, stream=True)
total = 0
with open(zip_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
        total += len(chunk)
        if total % (1024*1024*10) == 0:
            print(f"  Downloaded {total // (1024*1024)} MB...")

print("📦 Extracting...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall("/content/slake/")

print(f"✅ Images extracted: {len(os.listdir('/content/slake/imgs'))}")

# Now save the JSON splits
slake = load_dataset("BoKelvin/SLAKE")

def save_slake_split(split_data, json_path, split_name):
    data = []
    skipped = 0
    for item in split_data:
        if item.get("q_lang", "en") != "en":
            continue

        img_name = item["img_name"]  # e.g. "xmlab1/source.jpg"
        img_path = f"/content/slake/imgs/{img_name}"

        if not os.path.exists(img_path):
            skipped += 1
            continue

        answer = str(item["answer"]).strip()
        answer_type = str(item["answer_type"]).lower()
        if answer_type in ("closed", "yes/no"):
            answer_type = "yes/no"
        else:
            answer_type = "open"

        data.append({
            "image_name":  img_name,
            "question":    item["question"],
            "answer":      answer,
            "answer_type": answer_type,
            "q_lang":      "en"
        })

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ {split_name}: {len(data)} saved, {skipped} skipped (missing image)")
    return len(data)

n_train = save_slake_split(slake["train"],      "/content/slake/train.json",    "train")
n_val   = save_slake_split(slake["validation"], "/content/slake/validate.json", "val")
n_test  = save_slake_split(slake["test"],       "/content/slake/test.json",     "test")

print(f"\n✅ SLAKE complete: {n_train} train | {n_val} val | {n_test} test")

# Sanity check
with open("/content/slake/train.json") as f:
    sample = json.load(f)[0]
print(f"   Sample: {sample}")
