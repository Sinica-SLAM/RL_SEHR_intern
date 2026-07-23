import json
from tqdm import tqdm

def convert_msg(messages):
    """Convert original roles to LlamaFactory roles.
    assistant (counselor) -> gpt
    user (client)         -> human
    """
    new_messages = []
    
    for msg in messages:
        if msg["role"] == "assistant":
            new_messages.append({
                "from": "gpt",
                "value": msg["content"]
            })
        elif msg["role"] == "user":
            new_messages.append({
                "from": "human",
                "value": msg["content"]
            })

    return new_messages

def convert_msg_r(messages):
    """Convert original roles to LlamaFactory roles.
    assistant (counselor) -> human
    user (client)         -> gpt
    """
    new_messages = []
    for msg in messages:
        if msg["role"] == "assistant":
            new_messages.append({
                "from": "human",
                "value": msg["content"]
            })
        elif msg["role"] == "user":
            new_messages.append({
                "from": "gpt",
                "value": msg["content"]
            })
    return new_messages

def to_ppo_context(sample):
    
    chosen_sample = {
        "conversations": sample["messages"] + sample["positive"]
    }   

    rejected_sample = {
        "conversations": sample["messages"] + sample["negative"]
    }
    return chosen_sample, rejected_sample

def process_master_dataset(sample, sft_dataset, ppo_dataset):

    base_messages = sample["messages"] 
    
    system_text = (
        "You are an AI emotional support peer counselor. Keep your responses natural, "
        "deeply empathetic, and engagingly conversational. Provide a substantive, thoughtful, and fully developed reply."
        "You are encouraged to weave multiple emotional support strategies together."
    )
    
    # ==========================================
    # 1. 產生 SFT 資料集 (對話必須以 assistant 結尾)
    # ==========================================
    
    instruct = {
        "from": "system",
        "value": system_text
    }
        
    sft_sample_pos = {
        "conversations": [instruct] + convert_msg(base_messages[1:]) + convert_msg(sample["positive"])
    }
    sft_dataset.append(sft_sample_pos)
    
    sft_sample_neg = {
        "conversations": [instruct] + convert_msg(base_messages[1:]) + convert_msg(sample["negative"])
    }
    sft_dataset.append(sft_sample_neg)
        
    # ==========================================
    # 2. 產生 PPO 資料集 (對話必須以 user 結尾)
    # ==========================================
    
    ppo_sample_pos = {
        "conversations": convert_msg(base_messages) + convert_msg([sample["positive"][0]])
    }
    ppo_dataset.append(ppo_sample_pos)
    
    ppo_sample_neg = {
        "conversations": convert_msg(base_messages) + convert_msg([sample["negative"][0]])
    }
    ppo_dataset.append(ppo_sample_neg)
    
    
def process_split(input_path, split_name):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reward_data = []
    sft_dataset = []
    ppo_dataset = []
    skipped     = 0

    for sample in tqdm(data, desc=f"Processing {split_name}"):
        context = convert_msg_r(sample["messages"])

        # validate: must start with human, end with human
        if not context:
            skipped += 1
            continue
        if context[0]["from"] != "human":
            skipped += 1
            continue
        if context[-1]["from"] != "human":
            skipped += 1
            continue

        # positive: take only the first (user) turn
        positive_turn = sample["positive"][0]
        if positive_turn["role"] != "user":
            skipped += 1
            continue

        # negative: take only the first (user) turn
        negative_turn = sample["negative"][0]
        if negative_turn["role"] != "user":
            skipped += 1
            continue

        reward_sample = {
            "conversations": context,
            "chosen": {
                "from": "gpt",
                "value": positive_turn["content"]
            },
            "rejected": {
                "from": "gpt",
                "value": negative_turn["content"]
            }
        }

        reward_data.append(reward_sample)
        
        process_master_dataset(sample, sft_dataset, ppo_dataset)

    # save
    rm_path  = f"./DATA/RM_{split_name}.json"
    sft_path = f"./DATA/SFT_{split_name}.json"
    ppo_path = f"./DATA/PPO_{split_name}.json"

    with open(rm_path, "w", encoding="utf-8") as f:
        json.dump(reward_data, f, indent=2, ensure_ascii=False)
        
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(sft_dataset, f, indent=2, ensure_ascii=False)

    with open(ppo_path, "w", encoding="utf-8") as f:
        json.dump(ppo_dataset, f, indent=2, ensure_ascii=False)

    print(f"\n[{split_name}]")
    print(f"  Reward data : {len(reward_data)}")
    print(f"  SFT data    : {len(sft_dataset)}")
    print(f"  PPO data    : {len(ppo_dataset)}")
    print(f"  Skipped     : {skipped}")

    return len(reward_data), len(sft_dataset), len(ppo_dataset), skipped

# ── Run all splits ─────────────────────────────────────────────
splits = [
    ("./DATA/RL_generated_trainset(low).json", "trainset"),
    ("./DATA/RL_generated_validset(low).json", "validset"),
    ("./DATA/RL_generated_testset(low).json",  "testset"),
]

total_reward, total_SFT, total_PPO, total_skipped = 0, 0, 0, 0
for input_path, split_name in splits:
    r, s, p, x = process_split(input_path, split_name)
    total_reward   += r
    total_SFT   += s
    total_PPO   += p
    total_skipped  += x

print(f"\n{'='*40}")
print(f"  Total reward samples : {total_reward}")
print(f"  Total SFT samples : {total_SFT}")
print(f"  Total PPO samples : {total_PPO}")
print(f"  Total skipped        : {total_skipped}")
print(f"{'='*40}")