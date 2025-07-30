import json
import os

# Paths
input_path = "../data/code_alpaca_20k.json"
output_path = "../data/final_coding_dataset.jsonl"

# Make sure output folder exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load dataset
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Format into prompt-completion pairs
processed = []
for example in data:
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output_text = example.get("output", "").strip()

    if instruction and output_text:
        prompt = instruction
        if input_text:
            prompt += "\n\n" + input_text

        processed.append({
            "prompt": prompt,
            "completion": output_text
        })

# Save in JSONL format
with open(output_path, "w", encoding="utf-8") as f:
    for item in processed:
        json.dump(item, f)
        f.write("\n")

print(f"Preprocessing complete. Total examples: {len(processed)}")
print(f"Saved to: {output_path}")

