# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:28:03 2026

@author: Clio
"""

# Step 1. Load ESConv dataset
import json
import random
from collections import deque, defaultdict
import os

def merge_turns(dialog):

    merged = []
    last_speaker = None

    for turn in dialog:

        speaker = turn["speaker"]
        content = turn["content"].strip()

        if last_speaker == speaker:
            merged[-1]["content"] += " " + content
            
            # 保留最後 annotation
            merged[-1]["annotation"] = turn.get("annotation", {})

        else:
            merged.append({
                "speaker": speaker,
                "content": content,
                "annotation": turn.get("annotation", {})
            })

        last_speaker = speaker

    return merged

data = json.load(open("./DATA/ESConv.json"))

samples = []

context_window = 2

for dialog_id, dialog in enumerate(data):

    merged_dialog = merge_turns(dialog["dialog"])

    history = deque(maxlen=context_window * 2)

    last_supporter_idx = None

    for turn in merged_dialog:

        speaker = turn["speaker"]
        content = turn["content"]
        annotation = turn.get("annotation", {})

        # seeker
        if speaker == "seeker":

            history.append({
                "role": "user",
                "content": content
            })

            # feedback 給上一個 supporter
            feedback = annotation.get("feedback", None)

            if feedback is not None and last_supporter_idx is not None:
                samples[last_supporter_idx]["feedback"] = feedback

        # supporter
        elif speaker == "supporter":
            
            if len(history) != 0:

                strategy = annotation.get("strategy", None)
                        
                samples.append({
                    "dialog_id": dialog_id,
                    "messages": list(history),
                    "response": content,
                    "strategy": strategy,
                    "feedback": None,
                    "emotion": dialog.get("emotion_type", None),
                    "problem": dialog.get("problem_type", None)
                })
                
            last_supporter_idx = len(samples) - 1

            history.append({
                "role": "assistant",
                "content": f"[{strategy}] {content}"
            })
            
# 輸出 JSON
os.makedirs("./DATA", exist_ok=True)
output_path = "./DATA/esconv_processed.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f"Saved {len(samples)} samples to {output_path}")

# dialog-level grouping
dialog_groups = defaultdict(list)

for sample in samples:
    dialog_groups[sample["dialog_id"]].append(sample)

dialog_ids = list(dialog_groups.keys())

# shuffle
random.seed(42)
random.shuffle(dialog_ids)

# split ratio
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

n = len(dialog_ids)

train_end = int(n * train_ratio)
valid_end = train_end + int(n * valid_ratio)

train_ids = dialog_ids[:train_end]
valid_ids = dialog_ids[train_end:valid_end]
test_ids = dialog_ids[valid_end:]

train_data = []
valid_data = []
test_data = []

for did in train_ids:
    train_data.extend(dialog_groups[did])

for did in valid_ids:
    valid_data.extend(dialog_groups[did])

for did in test_ids:
    test_data.extend(dialog_groups[did])

# 輸出 JSON
with open("./DATA/esconv_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open("./DATA/esconv_valid.json", "w", encoding="utf-8") as f:
    json.dump(valid_data, f, indent=2, ensure_ascii=False)

with open("./DATA/esconv_test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print("Train:", len(train_data))
print("Valid:", len(valid_data))
print("Test :", len(test_data))