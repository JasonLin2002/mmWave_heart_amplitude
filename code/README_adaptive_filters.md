# 深度自適應濾波器系統使用說明

## 概述

本系統包含兩個主要程式：
1. `deep_adaptive_filter_trainer.py` - 深度學習模型訓練程式
2. `adaptive_filters_comparison.py` - 自適應濾波器比較程式

## 使用流程

### 步驟 1: 訓練深度學習模型

首先執行訓練程式來創建預訓練模型：

```bash
python deep_adaptive_filter_trainer.py
```

**訓練選項：**
- 選項 1：完整訓練（所有距離和模型類型）
- 選項 2：快速訓練（僅45cm hybrid模型） - **推薦新手使用**

### 步驟 2: 執行濾波器比較

訓練完成後，執行比較程式：

```bash
python adaptive_filters_comparison.py
```

## 程式功能說明

### deep_adaptive_filter_trainer.py

**主要功能：**
- 自動載入 mmWave 和 ECG 配對數據
- 支援多種深度學習模型架構：
  - **CNN模型** - 卷積神經網路
  - **LSTM模型** - 長短期記憶網路
  - **Hybrid模型** - CNN+LSTM混合架構 ⭐ **推薦**
  - **Transformer模型** - 自注意力機制

**訓練特色：**
- 自動數據預處理和重採樣
- GPU 自動檢測和使用
- 早停機制防止過擬合
- 自動保存最佳模型
- 訓練歷史視覺化

### adaptive_filters_comparison.py

**主要功能：**
- 整合 5 種濾波方法：
  1. **原始訊號** - 未處理的 mmWave 信號
  2. **LMS** - 最小均方自適應濾波器
  3. **RLS** - 遞迴最小平方自適應濾波器
  4. **NLMS** - 正規化 LMS 自適應濾波器
  5. **深度自適應濾波器 (DAF)** - 載入預訓練模型 🚀

**智慧功能：**
- 自動檢測並載入預訓練模型
- 如果預訓練模型不存在，自動回退到即時訓練
- 多層備用方案確保程式穩定運行

## 目錄結構

```
mmWave_heart_amplitude/code/
├── deep_adaptive_filter_trainer.py    # 深度學習訓練程式
├── adaptive_filters_comparison.py     # 濾波器比較程式
├── trained_models/                    # 預訓練模型目錄 (自動創建)
│   ├── daf_hybrid_45cm.h5            # 45cm 混合模型
│   ├── daf_cnn_45cm.h5               # 45cm CNN模型
│   └── ...                           # 其他訓練好的模型
├── training_history.png               # 訓練歷史圖表
├── Adaptive_Filters_ECG_Comparison.png # 濾波器比較結果圖
└── README_adaptive_filters.md         # 本說明文件
```

## 數據路徑設定

請確認以下路徑正確：

```python
# mmWave 數據路徑
mmwave_base_path = "mmWave-PAPER/Output/merged_csv"

# ECG 數據路徑  
ecg_base_path = "mmWave-PAPER/Output/ECG Data/CSV"
```

**支援的距離：**
- 30cm, 45cm, 60cm, 90cm

## 模型效能說明

### 模型比較

| 模型類型 | 訓練時間 | 記憶體使用 | 濾波效果 | 推薦程度 |
|---------|---------|-----------|---------|---------|
| CNN     | 快      | 低        | 良好    | ⭐⭐⭐   |
| LSTM    | 中      | 中        | 優秀    | ⭐⭐⭐⭐ |
| Hybrid  | 慢      | 高        | 最佳    | ⭐⭐⭐⭐⭐|
| Transformer | 最慢 | 最高      | 優秀    | ⭐⭐⭐⭐ |

### 性能優化建議

1. **GPU 加速**：如果有 NVIDIA GPU，將自動啟用 CUDA 加速
2. **記憶體管理**：訓練時會自動設定動態記憶體分配
3. **批次大小調整**：根據 GPU 記憶體調整 `batch_size`
4. **檔案數量限制**：透過 `max_pairs_per_distance` 控制訓練數據量

## 故障排除

### 常見問題

**1. 找不到數據檔案**
```
❌ mmWave檔案不存在 或 ECG檔案不存在
```
**解決方案**：檢查數據路徑是否正確

**2. GPU 記憶體不足**
```
ResourceExhaustedError: OOM when allocating tensor
```
**解決方案**：減少 `batch_size` 或 `max_pairs_per_distance`

**3. 預訓練模型載入失敗**
```
❌ 載入預訓練模型失敗
```
**解決方案**：程式會自動使用備用方案，無需擔心

### 效能調優

**快速訓練設定（推薦新手）：**
```python
distances = ['45cm']                    # 僅訓練一個距離
model_types = ['hybrid']                # 僅訓練混合模型
max_pairs_per_distance = 15             # 減少訓練數據
epochs = 30                             # 減少訓練輪數
```

**完整訓練設定（推薦進階使用者）：**
```python
distances = ['30cm', '45cm', '60cm', '90cm']  # 所有距離
model_types = ['hybrid', 'cnn', 'lstm']       # 多種模型
max_pairs_per_distance = 30                   # 更多訓練數據
epochs = 100                                  # 更多訓練輪數
```

## 輸出結果解讀

### 訓練輸出
- **Loss 曲線**：越低越好，應該持續下降
- **MAE 曲線**：平均絕對誤差，越小越好
- **驗證曲線**：應該與訓練曲線相近（避免過擬合）

### 比較結果
程式會輸出詳細的性能比較表格：
- **DTW距離**：動態時間規整距離（越小越好）
- **頻譜相干性**：頻域相似性（越大越好）
- **樣本熵相似性**：非線性相似性（越大越好）
- **綜合表現分數**：加權平均分數（越大越好）

## 技術支援

如果遇到問題，請檢查：
1. Python 環境是否安裝所需套件
2. TensorFlow 版本是否相容
3. 數據檔案路徑是否正確
4. GPU 驅動是否正確安裝（如果使用 GPU）

## 更新日誌

- **v1.0** - 基本自適應濾波器實作
- **v2.0** - 加入深度學習預訓練模型支援
- **v2.1** - 新增多種模型架構和自動備用方案

---

**開發者**: 深度自適應濾波器研究團隊  
**最後更新**: 2025/8/28
