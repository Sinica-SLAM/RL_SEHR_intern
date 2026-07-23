import json
import os
import re
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.optim import AdamW

# ==========================================
# 1. 參數配置
# ==========================================
MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
TRAIN_DATA_PATH = "DATA/SFT_trainset.json"  # 請依實際路徑調整
VALID_DATA_PATH = "DATA/SFT_validset.json"
OUTPUT_DIR = "models/policy_model_multi/"
BEST_OUTPUT_DIR = "models/policy_model_multi_best/"  # validation loss 最佳的那個 epoch，另外存一份
CUTOFF_LEN = 512            # 只作為「截斷上限」，batch 內仍採動態 padding
BATCH_SIZE = 8
GRAD_ACC = 8
EPOCHS = 10
LR = 3e-5                   # 全參數微調，維持與舊版接近的較小學習率
WARMUP_RATIO = 0.03         # 用比例算 warmup steps，取代寫死的 100 步
STRATEGY_LOSS_WEIGHT = 0.3  # 輔助分類任務降權，避免蓋過主要的生成 LM loss
USE_GRADIENT_CHECKPOINTING = True  # 全參數微調記憶體吃緊時開啟
EARLY_STOPPING_PATIENCE = 3  # 連續幾個 epoch validation loss 沒有改善就提前停止；設 None 關閉
MIN_DELTA = 1e-4             # 小於這個幅度的下降不算「有改善」，避免雜訊觸發誤判

STRATEGIES = [
    "[Affirmation and Reassurance]", "[Information]", "[Others]", "[Providing Suggestions]",
    "[Question]", "[Reflection of feelings]", "[Restatement or Paraphrasing]", "[Self-disclosure]"
]
strat2id = {strat: idx for idx, strat in enumerate(STRATEGIES)}

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

STRATEGY_ALIASES = {
    "[Strategy: Question]": "[Question]",
    "[Strategy: Reflection of feelings]": "[Reflection of feelings]",
    "[Strategy: Providing Suggestions]": "[Providing Suggestions]",
    "[Strategy: Affirmation and Reassurance]": "[Affirmation and Reassurance]",
    "[Strategy: Information]": "[Information]",
    "[Strategy: Self-disclosure]": "[Self-disclosure]",
    "[Strategy: Restatement or Paraphrasing]": "[Restatement or Paraphrasing]",
    "[Strategy: Questions]": "[Question]",
    "[Questions]": "[Question]",
    "[Strategy: Apology and Reassurance]": "[Affirmation and Reassurance]",
}

_unknown_label_counts = {}


def normalize_strategy_label(label: str) -> str:
    label = label.strip()
    if label in STRATEGY_ALIASES:
        return STRATEGY_ALIASES[label]
    if label.startswith("[Strategy: ") and label.endswith("]"):
        cleaned = label[len("[Strategy: "):-1].strip()
        label = f"[{cleaned}]"
        if label in STRATEGY_ALIASES:
            return STRATEGY_ALIASES[label]
    if label in STRATEGIES:
        return label
    
    _unknown_label_counts[label] = _unknown_label_counts.get(label, 0) + 1
    return "[Others]"

# ==========================================
# 2. 數據集定義 (完全適配 ShareGPT 格式)
# 注意：__getitem__ 不再手動 pad 到固定長度，改由 collate_fn 依 batch 內最長樣本動態 padding，
# 避免每筆樣本都被灌到 CUTOFF_LEN，浪費大量算力。
# ==========================================
class MultiTaskDialogDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=512):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        # 💡 速度優化：原本每次 __getitem__ 都重新 tokenize，10 個 epoch 就重複 tokenize 10 次。
        # ESConv 這種資料量不大（約千來筆對話），一次性在 init 時全部 tokenize 好、存進記憶體，
        # 之後每個 epoch 的 __getitem__ 都只是查表，CPU-side 開銷大幅降低（尤其在 num_workers 不夠多時很有感）。
        self.features = [self._build_feature(i) for i in range(len(self.data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx]

    def _build_feature(self, idx):
        # 1. 取得該樣本的對話歷史與最後一輪回覆
        item = self.data[idx]
        conversations = item["conversations"]

        # 分離最後一輪的模型回覆 (gpt) 及其前面的所有上下文
        history_turns = conversations[:-1]
        gpt_turn = conversations[-1]

        # 2. 建構 LLaMA 3 Chat Template 格式的 Prompt
        prompt = "<|begin_of_text|>"
        for turn in history_turns:
            role = turn["from"]
            # 將 ShareGPT 的角色對齊到 LLaMA 3 標頭
            if role == "human":
                role_id = "user"
            elif role == "system":
                role_id = "system"
            else:
                role_id = "assistant"

            prompt += f"<|start_header_id|>{role_id}<|end_header_id|>\n\n{turn['value']}<|eot_id|>"

        # 加上最後準備讓 Assistant 回覆的引導標頭
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        # 3. 處理 GPT 回覆文字與抽取策略標籤
        full_gpt_value = gpt_turn["value"].strip()

        # 正則表達式：精準抽取開頭的 [Strategy] 標籤
        match = re.match(r"^(\[.*?\])\s*(.*)$", full_gpt_value, re.DOTALL)
        if match:
            gold_strategy = normalize_strategy_label(match.group(1))
            gold_response = match.group(2)
        else:
            gold_strategy = "[Others]"
            gold_response = full_gpt_value

        target = f" {gold_strategy} {gold_response}<|eot_id|>"
        # 4. 轉換為 Token IDs
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        # 5. 只做截斷，不做 padding（padding 留給 collate_fn 依 batch 動態處理）
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]

        attention_mask = [1] * len(input_ids)

        # 6. 取得對應的分類 ID 與策略 Token 的精準索引位置
        strat_id = strat2id.get(gold_strategy, strat2id["[Others]"])

        # 💡 修正對齊問題：
        # hidden_states[:, i, :] 在因果模型中是用來「預測第 i+1 個 token」的表示，
        # 也就是模型「還沒看到」策略標籤本身、但context已經讀完的那個狀態。
        # 策略標籤的第一個 token 落在 input_ids 的 len(prompt_ids) 位置，
        # 所以要拿來預測它的 hidden state 應該取 len(prompt_ids) - 1，而不是 len(prompt_ids)。
        strategy_token_idx = len(prompt_ids) - 1
        strategy_token_idx = max(0, min(strategy_token_idx, len(input_ids) - 1))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "strat_id": strat_id,
            "strategy_token_idx": strategy_token_idx,
        }


def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
        labels = [torch.tensor(b["labels"], dtype=torch.long) for b in batch]
        attention_mask = [torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        strat_id = torch.tensor([b["strat_id"] for b in batch], dtype=torch.long)
        strategy_token_idx = torch.tensor([b["strategy_token_idx"] for b in batch], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "strat_id": strat_id,
            "strategy_token_idx": strategy_token_idx,
        }
    return collate_fn

# ==========================================
# 3. 核心：自訂多任務雙頭模型 (Multi-Task Model, 全參數微調)
# ==========================================
class MultiTaskLlama(nn.Module):
    def __init__(self, base_model, num_strategies):
        super().__init__()
        self.llama = base_model
        # 分類頭：吃 LLaMA 的隱藏維度，輸出策略類別的機率
        hidden_size = base_model.config.hidden_size
        classifier_dtype = next(base_model.parameters()).dtype
        self.strategy_classifier = nn.Linear(hidden_size, num_strategies, dtype=classifier_dtype)
        nn.init.normal_(self.strategy_classifier.weight, std=0.02)
        nn.init.zeros_(self.strategy_classifier.bias)

    def forward(self, input_ids, attention_mask, labels=None, strat_id=None, strategy_token_idx=None):
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        lm_loss = outputs.loss  # Task 1：生成 Loss (Causal LM Loss)

        last_hidden_state = outputs.hidden_states[-1]  # [Batch, SeqLen, HiddenSize]
        batch_size = input_ids.shape[0]

        strat_features = last_hidden_state[torch.arange(batch_size), strategy_token_idx]  # [Batch, HiddenSize]
        strat_features = strat_features.to(self.strategy_classifier.weight.dtype)
        strat_logits = self.strategy_classifier(strat_features)  # [Batch, NumStrategies]

        loss_fct = nn.CrossEntropyLoss()
        strat_loss = loss_fct(strat_logits, strat_id)

        return lm_loss, strat_loss, strat_logits

# ==========================================
# 4. Validation 與 checkpoint 工具
# ==========================================
def evaluate(model, valid_loader, device, distributed):
    """在 validation set 上算平均 lm_loss / strat_loss，DDP 下會把各 rank 的總和 all_reduce 起來平均。"""
    model.eval()
    sum_lm_loss = torch.zeros(1, device=device)
    sum_strat_loss = torch.zeros(1, device=device)
    sum_count = torch.zeros(1, device=device)

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            strat_id = batch["strat_id"].to(device)
            strategy_token_idx = batch["strategy_token_idx"].to(device)

            lm_loss, strat_loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                strat_id=strat_id,
                strategy_token_idx=strategy_token_idx,
            )
            bs = input_ids.size(0)
            sum_lm_loss += lm_loss.detach() * bs
            sum_strat_loss += strat_loss.detach() * bs
            sum_count += bs

    if distributed:
        dist.all_reduce(sum_lm_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_strat_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_count, op=dist.ReduceOp.SUM)

    model.train()
    avg_lm_loss = (sum_lm_loss / sum_count).item()
    avg_strat_loss = (sum_strat_loss / sum_count).item()
    return avg_lm_loss, avg_strat_loss


def save_checkpoint(unwrapped_model, tokenizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    unwrapped_model.llama.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(unwrapped_model.strategy_classifier.state_dict(), f"{save_dir}/strategy_head.pt")


# ==========================================
# 5. 訓練主程式
# ==========================================
def main():
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA GPUs.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    is_main_process = rank == 0
    if is_main_process:
        print(
            f"Using {'DDP on ' + str(dist.get_world_size()) + ' GPUs' if distributed else 'one device'}: {device}"
        )

    # 載入 Tokenizer 並手動將策略加為 Special Tokens
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": STRATEGIES})

    # 載入 Base Model 並 Resize Embedding（全參數微調，不套用 LoRA）
    if is_main_process:
        print("Loading base model (full fine-tuning, no LoRA)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    base_model.resize_token_embeddings(len(tokenizer))
    # Llama-3.2-1B/3B 為 tie_word_embeddings=True，resize 後 embed_tokens 與 lm_head 仍應保持綁定，
    # 全參數微調下兩者本來就是同一份權重，不會有 LoRA modules_to_save 拆開 tie 的問題。
    if getattr(base_model.config, "tie_word_embeddings", False):
        base_model.tie_weights()

    if USE_GRADIENT_CHECKPOINTING:
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False

    # 封裝成我們的多任務模型（全部參數皆可訓練）
    model = MultiTaskLlama(base_model, len(STRATEGIES)).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    def build_generation_prompt(sample):
        prompt = "<|begin_of_text|>"
        for turn in sample["conversations"][:-1]:
            role = turn["from"]
            if role == "human":
                role_id = "user"
            elif role == "system":
                role_id = "system"
            else:
                role_id = "assistant"
            prompt += f"<|start_header_id|>{role_id}<|end_header_id|>\n\n{turn['value']}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    # 準備數據
    train_dataset = MultiTaskDialogDataset(TRAIN_DATA_PATH, tokenizer, CUTOFF_LEN)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,            # 平行處理 batch 組裝，避免 GPU 等 CPU
        pin_memory=True,          # 加速 host->device 傳輸
        persistent_workers=True,  # 避免每個 epoch 重新 fork worker 的開銷
    )

    valid_dataset = MultiTaskDialogDataset(VALID_DATA_PATH, tokenizer, CUTOFF_LEN)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False) if distributed else None
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=valid_sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    if is_main_process:
        print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")

    sample_item = train_dataset.data[0]
    sample_prompt = build_generation_prompt(sample_item)

    try:
        # fused AdamW 在 CUDA 上把 optimizer step 的多個 kernel 合併，減少 launch overhead，通常有感提速
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01, fused=torch.cuda.is_available())
    except TypeError:
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    updates_per_epoch = (len(train_loader) + GRAD_ACC - 1) // GRAD_ACC
    total_steps = updates_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    if is_main_process:
        print(f"Total optim steps: {total_steps}, warmup steps: {warmup_steps}")
        print("Start Multi-Task Full Fine-Tuning...")
    model.train()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        total_lm_loss, total_strat_loss, total_combined_loss = 0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process)):
            batches_in_this_update = min(
                GRAD_ACC,
                len(train_loader) - (step // GRAD_ACC) * GRAD_ACC,
            )

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            strat_id = batch["strat_id"].to(device)
            strategy_token_idx = batch["strategy_token_idx"].to(device)

            lm_loss, strat_loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                strat_id=strat_id,
                strategy_token_idx=strategy_token_idx,
            )

            combined_loss = lm_loss + (STRATEGY_LOSS_WEIGHT * strat_loss)
            combined_loss = combined_loss / batches_in_this_update
            combined_loss.backward()

            total_lm_loss += lm_loss.item()
            total_strat_loss += strat_loss.item()
            total_combined_loss += combined_loss.item() * batches_in_this_update

            if (step + 1) % GRAD_ACC == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if is_main_process:
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Gen LM Loss: {total_lm_loss/len(train_loader):.4f}")
            print(f"Strat Classifier Loss: {total_strat_loss/len(train_loader):.4f}")
            print(f"Combined Loss: {total_combined_loss/len(train_loader):.4f}\n")

        # Epoch 生成驗證
        model.eval()
        if distributed:
            dist.barrier()
        if is_main_process:
            unwrapped_model = model.module if distributed else model
            strategy_token_ids = set(tokenizer.convert_tokens_to_ids(STRATEGIES))
            with torch.no_grad():
                inputs = tokenizer(sample_prompt, return_tensors="pt", add_special_tokens=False).to(device)

                # 探針：teacher-forced 直接看 8 個策略 token 在該位置的機率分布，
                # 比 greedy 生成更早看出學習趨勢（不會被 argmax 贏家全拿掩蓋掉漸進的進步）
                probe_logits = unwrapped_model.llama(**inputs).logits[0, -1, :]
                strat_id_list = list(strategy_token_ids)
                probe_probs = torch.softmax(probe_logits[strat_id_list], dim=-1)
                print(f"[Epoch {epoch+1} strategy-token probe] "
                      + ", ".join(f"{STRATEGIES[i]}={p:.3f}"
                                  for i, p in zip(range(len(STRATEGIES)), probe_probs.tolist())))

                generated = unwrapped_model.llama.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen_ids = generated[0][inputs["input_ids"].shape[-1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            print(f"[Epoch {epoch+1} generation check] {gen_text}")
            # 用 token id 判斷，不用字元切片（原本 gen_text[:20] 對較長的策略名稱會誤判成沒偵測到）
            if any(tid.item() in strategy_token_ids for tid in gen_ids[:3]):
                print("✅ strategy prefix detected in generation output")
            else:
                print("⚠️ no strategy prefix detected in generation output (greedy decode)")
        if distributed:
            dist.barrier()

        # ---- Validation：算 validation loss，避免只看 train loss 誤判模型狀況（過擬合時 train loss 會一直降但 val loss 會回升）----
        val_lm_loss, val_strat_loss = evaluate(model, valid_loader, device, distributed)
        val_combined = val_lm_loss + (STRATEGY_LOSS_WEIGHT * val_strat_loss)

        should_stop = torch.zeros(1, device=device)
        if is_main_process:
            print(f"[Epoch {epoch+1}] Val LM Loss: {val_lm_loss:.4f} | "
                  f"Val Strat Loss: {val_strat_loss:.4f} | Val Combined: {val_combined:.4f}")

            if val_combined < best_val_loss - MIN_DELTA:
                best_val_loss = val_combined
                patience_counter = 0
                unwrapped_model = model.module if distributed else model
                save_checkpoint(unwrapped_model, tokenizer, BEST_OUTPUT_DIR)
                print(f"✅ New best val loss ({best_val_loss:.4f})，已存到 {BEST_OUTPUT_DIR}")
            else:
                patience_counter += 1
                print(f"⚠️ Val loss 沒有改善（{patience_counter}/{EARLY_STOPPING_PATIENCE if EARLY_STOPPING_PATIENCE else '∞'}）")
                if EARLY_STOPPING_PATIENCE is not None and patience_counter >= EARLY_STOPPING_PATIENCE:
                    print("⏹ 觸發 early stopping，停止訓練")
                    should_stop[0] = 1.0

        # 把是否要停止的決定廣播給所有 rank，避免只有 rank0 跳出迴圈、其他 rank 卡在下一輪的 barrier/all_reduce 上
        if distributed:
            dist.broadcast(should_stop, src=0)
        model.train()

        if should_stop.item() == 1.0:
            break

    # 儲存模型（全參數權重，非 adapter；這是「最後一個 epoch」的權重，跟 BEST_OUTPUT_DIR 的最佳 checkpoint 分開存）
    if is_main_process:
        unwrapped_model = model.module if distributed else model
        print(f"Saving last-epoch weights to {OUTPUT_DIR}...")
        save_checkpoint(unwrapped_model, tokenizer, OUTPUT_DIR)
        if _unknown_label_counts:
            print("⚠️ 資料集中出現以下未對應到已知策略的標籤（已全部歸類為 [Others]）：")
            for label, count in sorted(_unknown_label_counts.items(), key=lambda x: -x[1]):
                print(f"   {label}: {count} 次")
        print(f"✅ Training complete! Best val loss = {best_val_loss:.4f} (saved at {BEST_OUTPUT_DIR})")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
