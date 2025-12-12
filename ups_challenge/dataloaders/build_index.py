import json
import os
import pickle

input_path = "ups_challenge/dataloaders/lang_id_results.jsonl" 
output_path = "lid_index.pkl"

index = {}

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        tar_number = obj["tar_number"]             
        filename = os.path.basename(obj["filepath"])  
        lang = obj["prediction"]                  

        index[(tar_number, filename)] = lang

print(f"Built index with {len(index)} entries")

with open(output_path, "wb") as f:
    pickle.dump(index, f, protocol=4)

print(f"Saved to {output_path}")