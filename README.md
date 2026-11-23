# NameThatLLM

辨識文字由哪一個語言模型（18 類 LLM）生成的實驗專案，涵蓋 BGE / MPNet + LoRA 以及 Qwen-3-Embedding + TaskLoRA（Layer Attention / All-Token Attention / Hierarchical Layer-Token / CosFace / Twin Head）等架構。

## 檔案結構

### 共用工具模組
- `data_utils.py`：資料路徑搜尋、亂數種子固定、GPU 顯存清理、訓練/測試集讀取、`prompt_id` 分組切分與 Kaggle 提交檔輸出。
- `metrics_utils.py`：Balanced Accuracy、Log-Softmax / LogLoss、單頭與雙頭溫度縮放及融合比例校準。
- `report_utils.py`：8 層樓 Layer Attention 權重統計表與全 Token 注意力重要性排行榜報表。
- `qwen_utils.py`：Qwen-3-Embedding 骨幹載入（支援前 24 層截斷與單/雙 GPU 配置）、TaskLoRA 外掛層、批次斷詞、權重存取與 Cosine Warmup 學習率排程。
- `sentence_lora_utils.py`：SentenceTransformer（BGE / MPNet）搭配 PEFT LoRA 與 Transformer 節奏分類器之共用流程。
- `NameThatLLM_AblationStudy.py`：Kaggle Notebook 一鍵打包腳本，執行後可於目前目錄自動產生上述 5 個 `*_utils.py` 模組。

### 實驗主程式
- `NameThatLLM.py`：凍結 BGE + RhythmCNN 基線模型。
- `NameThatLLM_LoRA.py` / `NameThatLLM_LoRA_mpnet.py`：BGE / all-mpnet-base-v2 搭配 PEFT LoRA 與 Transformer 分類器。
- `NameThatLLM_Qwen_AllToken.py` / `NameThatLLM_Qwen_AllToken_v2.py`：Qwen-3-Embedding + TaskLoRA + 全 Token 注意力分類器。
- `NameThatLLM_Qwen_Attention.py` / `NameThatLLM_Qwen_Attention_v2.py`：Qwen-3-Embedding + TaskLoRA + 8 層樓 Gated Layer Attention。
- `NameThatLLM_Qwen_LayerToken.py` / `NameThatLLM_Qwen_LayerToken_CosFace.py` / `NameThatLLM_Qwen_LayerToken_v2.py`：階層式雙重注意力（8 層樓 + 全 Token）、Sub-center CosFace 與去主題雙頭架構。
