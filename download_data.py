
from datasets import load_dataset
import json

ds = load_dataset("basicv8vc/SimpleQA")

print(f"Available splits: {list(ds.keys())}")

# Save each split to the root directory
for split_name, split_data in ds.items():
    filename = f"./SimpleQA_{split_name}.jsonl"
    with open(filename, 'w') as f:
        for item in split_data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(split_data)} examples to {filename}")
