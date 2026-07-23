# RL_SEHR_intern

## 建立環境和套件
建立虛擬環境 python>=3.11

安裝LlamaFactory套件: 
https://github.com/hiyouga/LlamaFactory#getting-started

前往對應資料夾 (我是放在user_jeremychang8裡面)

`cd /mnt/md0/user_jeremychang8/RL-with-HRE/`

前處理ESConv資料集，切成train/valid/test

`python ESConv_preprocessing.py`

## 資料前處理
[非必要步驟]以ESConv為基底，使用GPT生成對話資料 (耗費運算資源)

`CUDA_VISIBLE_DEVICES=0 python GPT_data_generation.py --split train --level low`

根據資料集train/valid/test、GPT生成難度去修改指令，執行前修改 my_api_key，研究如何修改 PROMPT_TEMPLATE，生成檔案為 ./DATA/RL_generated_testset(low).json
