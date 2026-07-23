# -*- coding: utf-8 -*-

import re
import json
import torch
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load as load_metric
from sklearn.metrics import classification_report, accuracy_score
import os
import numpy as np
from peft import PeftConfig, PeftModel

import torch.nn as nn
from transformers import AutoConfig, LlamaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import pipeline
# 下載 NLTK 斷詞資源（用於 BLEU 與 Distinct 計算）
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==========================================
# 1. 參數與路徑配置
# ==========================================
TEST_SET_PATH = "./DATA/esconv_test.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSIFIER_MODEL_PATH = "models/policy_model/"
CLASSIFIER_DIR = "models/strategy_model/best_strategy_llama_multiclass"  # 存分類器 Head 的路徑

GENERATION_MODEL_PATH = "/mnt/md0/user_jeremychang8/RL-with-HRE/models/policy_model_multi/"
GENERATION_MODEL_NAME = os.path.basename(GENERATION_MODEL_PATH.rstrip('/'))
print(f"Using device: {DEVICE}")

def load_tokenizer(path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    except Exception as e:
        print(f"Fast tokenizer unavailable for {path}, falling back to slow tokenizer: {e}")
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    return tokenizer

class LlamaMultiClassClassifier(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.llama = LlamaModel.from_pretrained(model_path)
        for param in self.llama.parameters():
            param.requires_grad = False
        hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        with torch.no_grad():
            outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        sequence_lengths = torch.eq(attention_mask, 0).int().argmax(dim=-1) - 1
        sequence_lengths = sequence_lengths.masked_fill(sequence_lengths < 0, -1)
        batch_size = input_ids.shape[0]
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        pooled_output = pooled_output.to(next(self.classifier.parameters()).dtype)
        logits = self.classifier(pooled_output)
        return SequenceClassifierOutput(logits=logits)
    
    def load_head(self, save_dir):
        self.classifier.load_state_dict(torch.load(os.path.join(save_dir, "classification_head.pt")))


# ==========================================
# 2. 載入模型與 Tokenizer
# ==========================================
print("Loading tokenizers...")
classifier_tokenizer = load_tokenizer(CLASSIFIER_MODEL_PATH)
classifier_tokenizer.pad_token = classifier_tokenizer.eos_token
classifier_tokenizer.padding_side = "left"

generation_tokenizer = load_tokenizer(GENERATION_MODEL_PATH)
generation_tokenizer.pad_token = generation_tokenizer.eos_token
generation_tokenizer.padding_side = "left"

# 載入標籤對照表
with open(os.path.join(CLASSIFIER_DIR, "strategy_mapping.json"), "r") as f:
    mapping = json.load(f)
id2label = {int(k): v for k, v in mapping["id2label"].items()}
label2id = mapping["label2id"]
unique_strategies = sorted(list(label2id.keys()))
num_labels = len(unique_strategies)

# 初始化分類器與其專屬的 Tokenizer (Left padding)
clf_tokenizer = load_tokenizer(CLASSIFIER_DIR)
if clf_tokenizer.pad_token is None:
    clf_tokenizer.pad_token = clf_tokenizer.eos_token
clf_tokenizer.padding_side = "left"

classifier_model = LlamaMultiClassClassifier(CLASSIFIER_MODEL_PATH, num_labels=num_labels)
classifier_model.load_head(CLASSIFIER_DIR)
classifier_model.to(DEVICE)
classifier_model.eval()

print("✅ Model Head loaded successfully.\n")

print("Loading emotion classification pipeline...")
# 💡 載入你指定好的 GoEmotions 預訓練模型
finetuned_emotion_model = 'SamLowe/roberta-base-go_emotions'
emotion_classifier = pipeline(
    "text-classification", 
    model=finetuned_emotion_model, 
    top_k=1, 
    device=0 if DEVICE == "cuda" else -1  # pipeline 的 device 參數通常接收整數 ID 或 -1
)
print("✅ Emotion Classifier loaded successfully.\n")

# ==========================================
# 3. 輔助功能定義（分句、預測、策略提取）
# ==========================================
def extract_strategy(text):
    text = text.strip()
    match = re.search(r"\[(.*?)\]", text)
    if match:
        return f"[{match.group(1)}]"
    return "[None]"

def decode_policy_response(token_ids):
    """保留 policy 模型輸出的策略 special token，僅移除結束/補齊控制 token。"""
    # 策略標籤在訓練時透過 additional_special_tokens 加入 tokenizer；因此不可
    # 使用 skip_special_tokens=True，否則例如 [Question] 也會被一起略過。
    text = generation_tokenizer.decode(token_ids, skip_special_tokens=False)
    control_tokens = {
        generation_tokenizer.eos_token,
        generation_tokenizer.pad_token,
        "<|end_of_text|>",
        "<|eot_id|>",
    }
    for token in control_tokens:
        if token:
            text = text.replace(token, "")
    return text.strip()

def build_policy_prompt(messages):
    """完全對齊 train_policy_model.py 的 Llama 3 prompt 格式。"""
    prompt = "<|begin_of_text|>"
    for message in messages:
        role = message["role"]
        if role == "human":
            role_id = "user"
        elif role == "system":
            role_id = "system"
        else:
            role_id = role  # esconv_test.json 使用 user / assistant
        prompt += (
            f"<|start_header_id|>{role_id}<|end_header_id|>\n\n"
            f"{message['content']}<|eot_id|>"
        )
    return prompt + "<|start_header_id|>assistant<|end_header_id|>\n\n"

def split_into_sentences(text):
    sentences = re.split(r'([。？！\.?!]\s*)', text)
    cleaned_sentences = []
    for i in range(0, len(sentences)-1, 2):
        sent = sentences[i].strip()
        punct = sentences[i+1].strip()
        if sent:
            cleaned_sentences.append(sent + punct)
    if len(sentences) % 2 != 0 and sentences[-1].strip():
        cleaned_sentences.append(sentences[-1].strip())
    return [s for s in cleaned_sentences if len(s) > 2]

def predict_composite_strategies(assistant_response):
    sentences = split_into_sentences(assistant_response)
    if not sentences:
        return set()
        
    formatted_inputs = []
    # for sent in sentences:
    #     if user_context:
    #         formatted_inputs.append(f"User: {user_context} </s> Assistant: {sent}")
    #     else:
    #         formatted_inputs.append(f"Assistant: {sent}")
    
    for sent in sentences:
        formatted_inputs.append(sent)
            
    inputs = clf_tokenizer(formatted_inputs, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    predicted_strategies = set()
    with torch.no_grad():
        outputs = classifier_model(**inputs)
        logits = outputs.logits.cpu().numpy()
        pred_ids = np.argmax(logits, axis=-1)
        
        for p_id in pred_ids:
            predicted_strategies.add(id2label[p_id])
    return predicted_strategies

def get_primary_emotion(text):
    if not text.strip():
        return None
    try:
        # pipeline(top_k=1) 會回傳如: [[{'label': 'sadness', 'score': 0.85}]]
        result = emotion_classifier(text)
        return result[0][0]['label']
    except Exception as e:
        return None

def predict_response_emotions(assistant_response):
    sentences = split_into_sentences(assistant_response)
    if not sentences:
        return set()
    
    emotion_set = set()
    try:
        # 批次處理由於子句數量通常不多，可以直接餵給 pipeline
        results = emotion_classifier(sentences)
        for res in results:
            emotion_set.add(res[0]['label'])
    except Exception as e:
        pass
    return emotion_set

# ==========================================
# 4. 讀取測試集並準備評估資料
# ==========================================
print(f"Reading testset from {TEST_SET_PATH}...")
with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

prompts = []         
references = []      
true_strategies = [] 
gold_strategies_raw = []  # 💡 新增：用來儲存未包裝中括號的原始策略字串，例如 "Question"
user_contexts = []        # 💡 新增：用來儲存每一筆資料的最後一句 User 對話

SYSTEM_PROMPT = (
    "You are an AI emotional support peer counselor. Keep your responses natural, "
    "deeply empathetic, short, and conversational (under 3 sentences). Focus on reflection "
    "and open-ended questions. Avoid generic medical disclaimers."
)

for sample in test_data:
    
    history_messages = sample["messages"]
    gold_response = sample["response"]
    gold_strategy = sample["strategy"]

    gold_strategies_raw.append(gold_strategy) # 存下原始字串，方便後面多標籤比對
    
    # 💡 核心修改一：從對話歷史中提取最後一輪的 User 對話
    user_context = ""
    for msg in reversed(history_messages):
        if msg["role"] == "user":
            user_context = msg["content"]
            break
    user_contexts.append(user_context)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    # 不使用 tokenizer 的 chat template，確保與訓練時的字串逐字一致。
    prompt_str = build_policy_prompt(messages)
    
    formatted_strategy = f"[{gold_strategy}]"
    gold_target = f"{formatted_strategy} {gold_response}"
    
    prompts.append(prompt_str)
    references.append(gold_target)
    true_strategies.append(formatted_strategy)

print(f"✅ Prepared {len(prompts)} test samples.\n")


# ==========================================
# 5. 進行推理生成 (Inference)
# ==========================================
OUTPUT_TXT_PATH = f"./RESULTS/{GENERATION_MODEL_NAME}/evaluation_results.txt"
RESPONSES_TXT_PATH = f"./RESULTS/{GENERATION_MODEL_NAME}/generated_responses.txt"
generated_responses = []
pred_strategies_first_token = [] # 舊方法：只抓第一個 token

if os.path.exists(RESPONSES_TXT_PATH):
    print(f"🔄 Found existing generation cache at {RESPONSES_TXT_PATH}.")
    print("Reading past responses to save time...")
    with open(RESPONSES_TXT_PATH, "r", encoding="utf-8") as f_cached:
        cached_lines = [line.strip() for line in f_cached.readlines() if line.strip()]
    
    if len(cached_lines) == len(prompts):
        print(f"✅ Cache valid! Successfully loaded {len(cached_lines)} responses. Skipping model inference.\n")
        generated_responses = cached_lines
        for res in generated_responses:
            pred_strategies_first_token.append(extract_strategy(res))
    else:
        print(f"⚠️ Cache count mismatch (Cache: {len(cached_lines)}, Testset: {len(prompts)}).")
        cached_lines = None

if not generated_responses:
    print(f"🚀 No valid cache found. Auto-detecting model architecture from: {GENERATION_MODEL_PATH}")
    
    # 💡 偵測 1：檢查路徑下有沒有 LoRA 設定檔
    lora_config_path = os.path.join(GENERATION_MODEL_PATH, "adapter_config.json")
    is_lora = os.path.exists(lora_config_path)
    if is_lora:
        # ----------------------------------------------------
        # 情況 B：這是分離式的 LoRA Adapter
        # ----------------------------------------------------
        print("📊 [Mode: LoRA Adapter] Loading base model + adapter...")
        
        # policy_model_new 是 train_policy_model.py 儲存的 PEFT/LoRA adapter。
        # 由 adapter_config 讀取底座名稱，避免日後更換底座時路徑不一致。
        peft_config = PeftConfig.from_pretrained(GENERATION_MODEL_PATH)
        BASE_MODEL_NAME = peft_config.base_model_name_or_path
        model_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True
        )
        
        print(f"🔧 Resizing base model embeddings to match generation tokenizer length: {len(generation_tokenizer)}")
        base_model.resize_token_embeddings(len(generation_tokenizer), mean_resizing=False)
        
        model = PeftModel.from_pretrained(
            base_model,
            GENERATION_MODEL_PATH,
            is_trainable=False,
        ).to(DEVICE)
    else:
        print("📦 [Mode: Merged/Full Model] Loading single integrated model...")
        model = AutoModelForCausalLM.from_pretrained(
            GENERATION_MODEL_PATH,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None
        )
        if DEVICE == "cuda":
            model = model.to(DEVICE)

    model.eval()
    print("✅ General Model Loader finished successfully. Set to eval mode.\n")

    for prompt in tqdm(prompts, desc="Generating"):
        inputs = generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.85,          # 0.7~0.9 之間,ESC 對話不宜過度發散
                top_p=0.9,                 # nucleus sampling,搭配 temperature 使用
                top_k=50,                  # 保留機率較高的候選,避免長尾亂跳
                repetition_penalty=1.15,   # 保留你原本的設定,避免重複詞
                no_repeat_ngram_size=3,    # 額外防止 3-gram 級別的重複(常見於情緒支持對話的口語化回覆)
                pad_token_id=generation_tokenizer.pad_token_id,
                eos_token_id=generation_tokenizer.eos_token_id
            )
        gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response_text = decode_policy_response(gen_tokens)
  
        generated_responses.append(response_text)
        pred_strategies_first_token.append(extract_strategy(response_text))

    print(f"💾 Saving generated responses to cache: {RESPONSES_TXT_PATH}")
    os.makedirs(os.path.dirname(RESPONSES_TXT_PATH), exist_ok=True)
    with open(RESPONSES_TXT_PATH, "w", encoding="utf-8") as f_pure:
        for res in generated_responses:
            f_pure.write(f"{res}\n")

# 6. 對每個生成回覆做分句策略／情緒預測
# ==========================================
print("Running segment-based strategy and emotion classification on generated responses...")
pred_strategy_sets = []
pred_emotion_sets = []

for res in tqdm(generated_responses, desc="Classifying Segments"):
    # 先把開頭的特殊標籤（例如 [Question]）去除，確保分類器只看對話內容
    cleaned_res = re.sub(r'\[.*?\]|<.*?>', '', res).strip()
    
    # 每個函式都先 split_into_sentences，再彙整所有句子的預測標籤。
    pred_strategy_sets.append(predict_composite_strategies(cleaned_res))
    pred_emotion_sets.append(predict_response_emotions(cleaned_res))


# ==========================================
# 7. 計算各項評估指標 (Metrics)
# ==========================================
print("Calculating evaluation metrics...")

# --- A. BLEU Score ---
bleu_metric = load_metric("bleu")
formatted_references = [[ref] for ref in references]

bleu1_results = bleu_metric.compute(predictions=generated_responses, references=formatted_references, max_order=1)
bleu2_results = bleu_metric.compute(predictions=generated_responses, references=formatted_references, max_order=2)

bleu_1 = bleu1_results["bleu"]
bleu_2 = bleu2_results["bleu"]

# --- B. BERTScore ---
try:
    bertscore_metric = load_metric("bertscore")
    bert_results = bertscore_metric.compute(
        predictions=generated_responses,
        references=references,
        lang="en",
        model_type="roberta-base"
    )
    mean_bert_f1 = sum(bert_results["f1"]) / len(bert_results["f1"]) if bert_results.get("f1") else 0.0
except Exception as e:
    print(f"Warning: BERTScore computation failed: {e}")
    mean_bert_f1 = 0.0

rouge_metric = load_metric("rouge")
# ROUGE 接收的格式與 BLEU 略有不同，references 只需要是一維的字串列表即可
rouge_results = rouge_metric.compute(predictions=generated_responses, references=references)
rouge_l = rouge_results["rougeL"]

# --- C. Distinct-N ---
def calculate_distinct_n(texts, n):
    total_ngrams = 0
    unique_ngrams = set()
    for text in texts:
        tokens = nltk.word_tokenize(text.lower())
        ngrams = list(nltk.ngrams(tokens, n))
        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0

distinct_1 = calculate_distinct_n(generated_responses, 1)
distinct_2 = calculate_distinct_n(generated_responses, 2)

# --- D. 💡 核心修改三：轉換為多標籤矩陣，計算複合策略的分類報告 ---
all_true_multihot = []
all_pred_multihot = []
strategy_hit_count = 0  
    
# 分句情緒共鳴度統計變數：回覆任一句命中 user 的主要情緒即算命中。
segment_emotion_hit_count = 0
valid_emotion_samples = 0 # 用來當分母（扣除 user_ctx 為空的情況）
user_primary_emotions = [None] * len(generated_responses)

print("Calculating Strategy and Emotion alignment metrics...")
for i in range(len(generated_responses)):
    gold_strat = gold_strategies_raw[i]
    pred_set = pred_strategy_sets[i]
    user_ctx = user_contexts[i]
    cleaned_res = re.sub(r'\[.*?\]|<.*?>', '', generated_responses[i]).strip()
    
    # === 1. 策略命中計算 ===
    true_vec = [1 if strat == gold_strat else 0 for strat in unique_strategies]
    pred_vec = [1 if strat in pred_set else 0 for strat in unique_strategies]
    all_true_multihot.append(true_vec)
    all_pred_multihot.append(pred_vec)
    
    if gold_strat in pred_set:
        strategy_hit_count += 1
        
    # === 2. 分句情緒共鳴命中計算 ===
    if user_ctx.strip():  # 確保 User 有輸入文字
        valid_emotion_samples += 1
        
        # 取得 User 的主要情緒 (e.g., 'sadness')
        user_emotion = get_primary_emotion(user_ctx)
        user_primary_emotions[i] = user_emotion
        
        # pred_emotion_sets 已由回覆的每一個 segment 彙整而成。
        response_emotions = pred_emotion_sets[i]
                
        # 如果 User 的情緒成功出現在任一回應 segment，判定正確。
        if user_emotion and (user_emotion in response_emotions):
            segment_emotion_hit_count += 1

# 計算專屬新指標
new_strategy_acc = strategy_hit_count / len(generated_responses)
# 計算 Segment Emotion Accuracy
segment_emotion_acc = (
    segment_emotion_hit_count / valid_emotion_samples if valid_emotion_samples > 0 else 0.0
)

# 產出分句分類的標準多標籤細部報告
segment_report_str = classification_report(
    np.array(all_true_multihot),
    np.array(all_pred_multihot),
    target_names=unique_strategies,
    zero_division=0
)
old_strategy_acc = accuracy_score(true_strategies, pred_strategies_first_token)

# ==========================================
# 8. 儲存結果並輸出至 TXT 檔案
# ==========================================
report_header = (
    "==================================================\n"
    "             POLICY MODEL EVALUATION              \n"
    "==================================================\n\n"
    f"Evaluated Model: {GENERATION_MODEL_PATH}\n"
    f"Total Test Samples: {len(prompts)}\n\n"
    "----- Quantitative Results -----\n"
    f"BLEU-1 Score         : {bleu_1:.4f}\n"
    f"BLEU-2 Score         : {bleu_2:.4f}\n"
    f"BERTScore (F1)     : {mean_bert_f1:.4f}\n"
    f"ROUGE-L            : {rouge_l:.4f}\n"
    f"Distinct-1         : {distinct_1:.4f}\n"
    f"Distinct-2         : {distinct_2:.4f}\n"
    f"Segment Emotion Acc : {segment_emotion_acc:.4f}\n"
    f"Strategy Acc       : {old_strategy_acc:.4f}\n"
    f"Segment Strategy Acc: {new_strategy_acc:.4f}\n\n"  # 
    "----- 💡 Segment-Based Multi-Label Strategy Report -----\n"
    f"{segment_report_str}\n"
)
print(report_header)

print(f"Writing results to {OUTPUT_TXT_PATH}...")
with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
    f.write(report_header)
    
    f.write("----- Case-by-Case Predictions -----\n")
    for i in range(len(prompts)):
        f.write(f"\n[Sample {i+1}]\n")
        f.write(f"True Strategy (Golden)  : {true_strategies[i]}\n")
        f.write(f"Old Pred (First Token)  : {pred_strategies_first_token[i]}\n")
        f.write(f"💡 New Pred (All Segments): {list(pred_strategy_sets[i])}\n")
        f.write(f"User Primary Emotion    : {user_primary_emotions[i]}\n")
        f.write(f"Pred Emotions (Segments): {list(pred_emotion_sets[i])}\n")
        f.write(f"Reference               : {references[i]}\n")
        f.write(f"Generated               : {generated_responses[i]}\n")
        f.write("-" * 40 + "\n")

print("🎉 Evaluation finished! Please check evaluation_results.txt.")
