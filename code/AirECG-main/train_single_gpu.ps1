# PowerShell 腳本用於單 GPU 訓練 AirECG 模型
$env:CUDA_VISIBLE_DEVICES="0"

# 切換到正確的目錄並執行單 GPU 訓練腳本
cd mmWave_heart_amplitude/code/AirECG-main
python train_single_gpu.py --global-batch-size 96
