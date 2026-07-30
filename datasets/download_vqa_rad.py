import os
import json
from datasets import load_dataset

# Create directories
os.makedirs("/content/vqa_rad/images", exist_ok=True)

# Load dataset
print("📥 Loading VQA-RAD from HuggingFace...")
vqarad = load_dataset("flaviagiammarino/vqa-rad")
print("Splits available:", vqarad.keys())
print("Columns:", vqarad["train"].column_names)

# Save function
def save_vqarad_split(split_data, json_path, split_name):
    data = []
    for idx, item in enumerate(split_data):
        img_name = f"vqarad_{split_name}_{idx}.jpg"
        img_path = f"/content/vqa_rad/images/{img_name}"
        try:
            item["image"].save(img_path)
        except Exception as e:
            print(f"  ⚠️ Image save failed at idx {idx}: {e}")
            continue
        answer = str(item["answer"]).strip()
        data.append({
            "image_name":  img_name,
            "question":    item["question"],
            "answer":      answer,
            "answer_type": "closed" if answer.lower() in ("yes", "no") else "open"
        })
        if idx % 500 == 0:
            print(f"  {split_name}: {idx} done...")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {len(data)} samples to {json_path}")
    return len(data)

# Save both splits
n_train = save_vqarad_split(
    vqarad["train"],
    "/content/vqa_rad/vqa_rad_train.json",
    "train"
)
n_test = save_vqarad_split(
    vqarad["test"],
    "/content/vqa_rad/vqa_rad_test.json",
    "test"
)

print(f"\n✅ VQA-RAD complete: {n_train} train | {n_test} test")
print(f"   Images: {len(os.listdir('/content/vqa_rad/images'))}")


with open("/content/vqa_rad/vqa_rad_train.json") as f:
    sample = json.load(f)[0]
print(f"   Sample: {sample}")
