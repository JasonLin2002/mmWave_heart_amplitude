# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from dtaidistance import dtw
from scipy.signal import coherence
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. 數據載入和預處理函數

# %%
def load_ecg_data(ecg_path):
    """載入並預處理ECG數據 (最後60秒)"""
    print(f"正在載入ECG數據: {ecg_path}")
    
    try:
        ecg_df = pd.read_csv(ecg_path)
        print(f"ECG數據形狀: {ecg_df.shape}")
        print("\nECG數據欄位:")
        print(ecg_df.columns.tolist())
        
        # 檢查必要的欄位是否存在
        if 'time' not in ecg_df.columns or 'ecg' not in ecg_df.columns:
            print("錯誤: ECG CSV 檔案中缺少 'time' 或 'ecg' 欄位。")
            return None, None, None
            
        # 移除 'ecg' 欄位中 NaN 的行，並確保 'time' 也相應移除
        df_cleaned = ecg_df.dropna(subset=['ecg'])
        
        ecg_signal = df_cleaned['ecg'].values
        timestamps_ns = df_cleaned['time'].values  # 時間戳（奈秒）
        
        if len(timestamps_ns) < 2:
            print("錯誤: ECG 數據不足以計算採樣率。")
            return None, None, None
        
        # ECG採樣率為130Hz
        ecg_fs = 130.0
        print(f"ECG 設定採樣率: {ecg_fs:.2f} Hz")
        
        # 轉換時間戳為秒並取最後60秒
        times_s = timestamps_ns * 1e-9
        N = len(ecg_signal)
        total_duration = times_s[-1] - times_s[0] if N > 1 else 0
        display_duration = 60.0
        
        if total_duration <= display_duration:
            start_idx = 0
            end_idx = N
        else:
            end_time = times_s[-1]
            start_time = end_time - display_duration
            start_idx = np.searchsorted(times_s, start_time, side="left")
            end_idx = N
            
        # 取最後60秒的數據
        ecg_signal = ecg_signal[start_idx:end_idx]
        timestamps_ns = timestamps_ns[start_idx:end_idx]
        times_s = times_s[start_idx:end_idx]
        
        # 創建相對時間軸（從0開始）
        ecg_time = times_s - times_s[0]
        
        print(f"ECG信號長度: {len(ecg_signal)} 樣本 ({len(ecg_signal)/ecg_fs:.1f} 秒)")
        print(f"ECG時間範圍: {ecg_time[0]:.1f} 到 {ecg_time[-1]:.1f} 秒")
        
        return ecg_signal, ecg_time, ecg_fs
        
    except Exception as e:
        print(f"載入ECG數據錯誤: {e}")
        return None, None, None

def load_and_preprocess_data(file_path):
    """讀取並預處理雷達心律振幅數據 (只保留最後60秒)"""
    print(f"讀取檔案: {file_path}")
    df = pd.read_csv(file_path)
    print(f"數據形狀: {df.shape}")
    print("\n數據預覽:")
    print(df.head())
    fs = 11.11  # 固定採樣率
    print(f"\n估計採樣率: {fs:.2f} Hz")
    df_sorted = df.sort_values(by='Frame_Number')
    waveform_data = df_sorted['Heart_Waveform'].values
    timestamps = df_sorted['Timestamp'].values
    heart_rates = df_sorted['Heart_Rate'].values
    frame_numbers = df_sorted['Frame_Number'].values

    # 只保留最後60秒
    N = len(waveform_data)
    time_axis_full = np.arange(N) / fs
    total_duration = time_axis_full[-1] if N > 1 else 0
    display_duration = 60.0
    if total_duration <= display_duration:
        start_idx = 0
        end_idx = N
    else:
        end_time = time_axis_full[-1]
        start_time = end_time - display_duration
        start_idx = np.searchsorted(time_axis_full, start_time, side="left")
        end_idx = N
    waveform_data = waveform_data[start_idx:end_idx]
    timestamps = timestamps[start_idx:end_idx]
    heart_rates = heart_rates[start_idx:end_idx]
    frame_numbers = frame_numbers[start_idx:end_idx]
    return waveform_data, timestamps, heart_rates, frame_numbers, fs

# %% [markdown]
# ## 2. 自適應濾波器實作

# %%
class LMSFilter:
    """最小均方 (LMS) 自適應濾波器"""
    def __init__(self, filter_length=32, step_size=0.01):
        self.filter_length = filter_length
        self.step_size = step_size
        self.weights = np.zeros(filter_length)
        self.input_buffer = np.zeros(filter_length)
        
    def filter(self, input_signal, desired_signal=None):
        """
        LMS濾波處理
        Args:
            input_signal: 輸入信號
            desired_signal: 期望信號 (如果為None，則使用延遲版本的輸入信號)
        """
        if desired_signal is None:
            # 使用延遲版本的輸入信號作為期望信號
            desired_signal = np.concatenate([np.zeros(10), input_signal[:-10]])
        
        output_signal = np.zeros(len(input_signal))
        
        for n in range(len(input_signal)):
            # 更新輸入緩衝區
            self.input_buffer = np.roll(self.input_buffer, 1)
            self.input_buffer[0] = input_signal[n]
            
            # 計算濾波器輸出
            y_n = np.dot(self.weights, self.input_buffer)
            output_signal[n] = y_n
            
            # 計算誤差
            error = desired_signal[n] - y_n
            
            # 更新權重 (LMS算法)
            self.weights += self.step_size * error * self.input_buffer
            
        return output_signal

class NLMSFilter:
    """正規化最小均方 (NLMS) 自適應濾波器"""
    def __init__(self, filter_length=32, step_size=0.1):
        self.filter_length = filter_length
        self.step_size = step_size
        self.weights = np.zeros(filter_length)
        self.input_buffer = np.zeros(filter_length)
        self.epsilon = 1e-10  # 避免除零
        
    def filter(self, input_signal, desired_signal=None):
        """
        NLMS濾波處理
        """
        if desired_signal is None:
            # 使用延遲版本的輸入信號作為期望信號
            desired_signal = np.concatenate([np.zeros(10), input_signal[:-10]])
        
        output_signal = np.zeros(len(input_signal))
        
        for n in range(len(input_signal)):
            # 更新輸入緩衝區
            self.input_buffer = np.roll(self.input_buffer, 1)
            self.input_buffer[0] = input_signal[n]
            
            # 計算濾波器輸出
            y_n = np.dot(self.weights, self.input_buffer)
            output_signal[n] = y_n
            
            # 計算誤差
            error = desired_signal[n] - y_n
            
            # 計算正規化因子
            norm_factor = np.dot(self.input_buffer, self.input_buffer) + self.epsilon
            
            # 更新權重 (NLMS算法)
            self.weights += (self.step_size / norm_factor) * error * self.input_buffer
            
        return output_signal

class RLSFilter:
    """遞迴最小平方 (RLS) 自適應濾波器"""
    def __init__(self, filter_length=32, forgetting_factor=0.99, delta=1.0):
        self.filter_length = filter_length
        self.lambda_ = forgetting_factor
        self.weights = np.zeros(filter_length)
        self.P = np.eye(filter_length) / delta  # 逆相關矩陣
        self.input_buffer = np.zeros(filter_length)
        
    def filter(self, input_signal, desired_signal=None):
        """
        RLS濾波處理
        """
        if desired_signal is None:
            # 使用延遲版本的輸入信號作為期望信號
            desired_signal = np.concatenate([np.zeros(10), input_signal[:-10]])
        
        output_signal = np.zeros(len(input_signal))
        
        for n in range(len(input_signal)):
            # 更新輸入緩衝區
            self.input_buffer = np.roll(self.input_buffer, 1)
            self.input_buffer[0] = input_signal[n]
            
            # 計算濾波器輸出
            y_n = np.dot(self.weights, self.input_buffer)
            output_signal[n] = y_n
            
            # 計算誤差
            error = desired_signal[n] - y_n
            
            # RLS更新
            k = self.P @ self.input_buffer
            alpha = 1.0 / (self.lambda_ + self.input_buffer @ k)
            k = alpha * k
            
            # 更新權重
            self.weights += k * error
            
            # 更新逆相關矩陣
            self.P = (self.P - np.outer(k, self.input_buffer @ self.P)) / self.lambda_
            
        return output_signal

class DeepAdaptiveFilter:
    """深度自適應濾波器 (DAF)"""
    def __init__(self, input_dim=32, hidden_units=64):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.model = None
        self.is_trained = False
        
    def build_model(self):
        """建構深度學習模型"""
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(self.hidden_units, activation='tanh'),
            layers.Dense(self.hidden_units//2, activation='tanh'),
            layers.Dense(self.hidden_units//4, activation='tanh'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
        
    def prepare_training_data(self, input_signal, desired_signal=None):
        """準備訓練數據"""
        if desired_signal is None:
            # 使用延遲版本的輸入信號作為期望信號
            desired_signal = np.concatenate([np.zeros(10), input_signal[:-10]])
            
        # 創建滑動窗口數據
        X = []
        y = []
        
        for i in range(self.input_dim, len(input_signal)):
            window = input_signal[i-self.input_dim:i]
            X.append(window)
            y.append(desired_signal[i])
            
        return np.array(X), np.array(y)
    
    def filter(self, input_signal, desired_signal=None):
        """
        深度自適應濾波處理
        """
        # 準備訓練數據
        X_train, y_train = self.prepare_training_data(input_signal, desired_signal)
        
        # 建構並訓練模型
        if not self.is_trained:
            self.model = self.build_model()
            
            # 訓練模型 (使用較少的epoch以減少計算時間)
            self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            self.is_trained = True
        
        # 生成輸出信號
        output_signal = np.zeros(len(input_signal))
        
        # 前面部分用零填充
        output_signal[:self.input_dim] = 0
        
        # 使用模型預測後面部分
        predictions = self.model.predict(X_train, verbose=0)
        output_signal[self.input_dim:] = predictions.flatten()
        
        return output_signal

# %% [markdown]
# ## 3. 自適應濾波處理函數

# %%
def apply_detrend(signal_data):
    """去趨勢處理"""
    return signal.detrend(signal_data)

def process_adaptive_filtering_methods(waveform_data, fs):
    """
    4種自適應濾波方法處理
    
    處理流程：
    1. 原始訊號
    2. LMS自適應濾波器
    3. RLS自適應濾波器  
    4. NLMS自適應濾波器
    5. 深度自適應濾波器 (DAF)
    """
    
    results = {}
    
    print("🔄 開始自適應濾波處理...")
    
    # 1. 原始訊號
    results['1_raw'] = waveform_data
    print("✅ 1. 原始訊號")
    
    # 基本前處理 - 去趨勢
    detrended = apply_detrend(waveform_data)
    
    # 正規化信號以改善濾波器性能
    normalized_signal = (detrended - np.mean(detrended)) / np.std(detrended)
    
    # 2. LMS自適應濾波器
    try:
        lms_filter = LMSFilter(filter_length=32, step_size=0.01)
        lms_output = lms_filter.filter(normalized_signal)
        results['2_lms'] = lms_output
        print("✅ 2. LMS自適應濾波器")
    except Exception as e:
        print(f"❌ LMS濾波器錯誤: {e}")
        results['2_lms'] = normalized_signal
    
    # 3. RLS自適應濾波器
    try:
        rls_filter = RLSFilter(filter_length=32, forgetting_factor=0.99)
        rls_output = rls_filter.filter(normalized_signal)
        results['3_rls'] = rls_output
        print("✅ 3. RLS自適應濾波器")
    except Exception as e:
        print(f"❌ RLS濾波器錯誤: {e}")
        results['3_rls'] = normalized_signal
    
    # 4. NLMS自適應濾波器
    try:
        nlms_filter = NLMSFilter(filter_length=32, step_size=0.1)
        nlms_output = nlms_filter.filter(normalized_signal)
        results['4_nlms'] = nlms_output
        print("✅ 4. NLMS自適應濾波器")
    except Exception as e:
        print(f"❌ NLMS濾波器錯誤: {e}")
        results['4_nlms'] = normalized_signal
    
    # 5. 深度自適應濾波器 (DAF) - 嘗試載入預訓練模型
    try:
        # 預訓練模型路徑（如果存在）
        pretrained_model_path = "mmWave_heart_amplitude/code/trained_models/daf_hybrid_45cm.h5"
        
        # 檢查是否存在預訓練模型
        if os.path.exists(pretrained_model_path):
            print(f"🔍 發現預訓練模型: {pretrained_model_path}")
            daf_filter = DeepAdaptiveFilter(input_dim=64, model_path=pretrained_model_path, model_type='hybrid')
        else:
            print("⚠️  未找到預訓練模型，將使用即時訓練")
            daf_filter = DeepAdaptiveFilter(input_dim=64, model_path=None, model_type='hybrid')
        
        daf_output = daf_filter.filter(normalized_signal)
        results['5_daf'] = daf_output
        print("✅ 5. 深度自適應濾波器 (DAF)")
    except Exception as e:
        print(f"❌ 深度自適應濾波器錯誤: {e}")
        results['5_daf'] = normalized_signal
    
    print("🎉 所有自適應濾波處理完成!")
    
    return results

# %% [markdown]
# ## 4. 視覺化函數

# %%
def visualize_adaptive_filtering_methods(results, fs, ecg_signal=None, ecg_time=None, output_filename=None):
    """可視化5種自適應濾波方法與ECG的比較"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互模式
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"字體設定錯誤: {e}")

    time = np.arange(len(results['1_raw'])) / fs
    
    # 定義5種方法的詳細名稱和顏色
    methods = [
        ('1_raw', '原始訊號 (Raw Signal)', 'blue'),
        ('2_lms', 'LMS自適應濾波器', 'red'),
        ('3_rls', 'RLS自適應濾波器', 'green'),
        ('4_nlms', 'NLMS自適應濾波器', 'orange'),
        ('5_daf', '深度自適應濾波器 (DAF)', 'purple')
    ]

    plt.figure(figsize=(20, 20))
    
    for i, (method_key, method_name, color) in enumerate(methods):
        if method_key in results:
            ax = plt.subplot(3, 2, i+1)
            ax.plot(time, results[method_key], color=color, linewidth=1.0, label=f'mmWave {method_name}')
            ax.set_title(f'{method_name} vs ECG', fontsize=14, fontweight='bold')
            ax.set_ylabel('mmWave振幅', color=color, fontsize=12)
            ax.tick_params(axis='y', labelcolor=color)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 60)
            
            # 添加ECG參考線
            if ecg_signal is not None and ecg_time is not None:
                ax_twin = ax.twinx()
                ax_twin.plot(ecg_time, ecg_signal, 'k-', alpha=0.6, linewidth=0.8, label='ECG參考')
                ax_twin.set_ylabel('ECG振幅', color='k', fontsize=12)
                ax_twin.tick_params(axis='y', labelcolor='k')
                
                # 添加圖例
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax_twin.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
            else:
                ax.legend(loc='upper right', fontsize=10)
            
            # 設置x軸刻度
            ax.set_xticks(np.arange(0, 61, 10))
            ax.set_xticklabels([str(x) for x in range(0, 61, 10)])
            
            if i >= 3:  # 下方的圖添加x軸標籤
                ax.set_xlabel('時間 (秒)', fontsize=12)

    plt.tight_layout()
    
    # 保存圖片
    if output_filename is None:
        output_filename = 'Adaptive_Filters_ECG_Comparison.png'
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"📊 自適應濾波器比較圖已儲存為: {output_filename}")
    plt.show()
    plt.close()

# %% [markdown]
# ## 5. 評估指標計算函數

# %%
def calculate_spectral_coherence(signal1, signal2, fs, nperseg=256):
    """計算兩個訊號之間的頻譜相干性 (頻域指標)"""
    try:
        # 計算相干性
        f, Cxy = coherence(signal1, signal2, fs, nperseg=nperseg)
        
        # 計算心律頻率範圍內 (0.8-3.0 Hz) 的平均相干性
        freq_mask = (f >= 0.8) & (f <= 3.0)
        if np.any(freq_mask):
            mean_coherence = np.mean(Cxy[freq_mask])
        else:
            mean_coherence = np.mean(Cxy)
            
        return mean_coherence
        
    except Exception as e:
        print(f"頻譜相干性計算錯誤: {e}")
        return 0.0

def calculate_sample_entropy(signal_data, m=2, r=None):
    """計算樣本熵 (非線性指標)"""
    try:
        # 正規化訊號
        signal_norm = (signal_data - np.mean(signal_data)) / np.std(signal_data)
        
        if r is None:
            r = 0.2 * np.std(signal_norm)
        
        N = len(signal_norm)
        
        def _maxdist(xi, xj, m):
            """計算兩個模式間的最大距離"""
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            """計算phi(m)"""
            patterns = np.array([signal_norm[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template, patterns[j], m) <= r:
                        C[i] += 1.0
                        
            phi = np.mean([np.log(c / (N - m + 1.0)) for c in C if c > 0])
            return phi
        
        # 計算樣本熵
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        if phi_m == 0 or phi_m1 == 0:
            return 0.0
            
        sample_entropy = phi_m - phi_m1
        
        return sample_entropy
        
    except Exception as e:
        print(f"樣本熵計算錯誤: {e}")
        return 0.0

def analyze_adaptive_filtering_performance(results, ecg_signal, ecg_time, mmwave_time, fs=11.11):
    """分析5種自適應濾波方法的性能"""
    print("\n" + "="*80)
    print("自適應濾波方法性能分析")
    print("="*80)
    
    # 標準化訊號以進行公平比較
    def normalize_signal(signal):
        return (signal - np.mean(signal)) / np.std(signal)
    
    # 重採樣ECG訊號到mmWave的時間軸
    ecg_resampled = np.interp(mmwave_time, ecg_time, ecg_signal)
    ecg_norm = normalize_signal(ecg_resampled)
    
    # 定義比較方法的詳細名稱
    method_names = {
        '1_raw': '1.原始訊號',
        '2_lms': '2.LMS自適應濾波器',
        '3_rls': '3.RLS自適應濾波器',
        '4_nlms': '4.NLMS自適應濾波器',
        '5_daf': '5.深度自適應濾波器(DAF)'
    }
    
    # 計算評估指標
    dtw_distances = {}           # 時間域指標
    spectral_coherences = {}     # 頻域指標  
    sample_entropies = {}        # 非線性指標
    composite_scores = {}        # 綜合表現分數
    
    for method_key, method_name in method_names.items():
        if method_key in results:
            signal_data = results[method_key]
            signal_norm = normalize_signal(signal_data)
            
            # 1. 時間域指標 - DTW距離計算
            try:
                dtw_distance = dtw.distance(signal_norm, ecg_norm)
                dtw_distances[method_key] = dtw_distance
            except Exception as e:
                print(f"DTW計算錯誤 ({method_name}): {e}")
                dtw_distances[method_key] = float('inf')
            
            # 2. 頻域指標 - 頻譜相干性計算
            try:
                coherence_val = calculate_spectral_coherence(signal_norm, ecg_norm, fs)
                spectral_coherences[method_key] = coherence_val
            except Exception as e:
                print(f"頻譜相干性計算錯誤 ({method_name}): {e}")
                spectral_coherences[method_key] = 0.0
            
            # 3. 非線性指標 - 樣本熵計算
            try:
                signal_entropy = calculate_sample_entropy(signal_norm)
                ecg_entropy = calculate_sample_entropy(ecg_norm)
                entropy_similarity = 1 / (1 + abs(signal_entropy - ecg_entropy))
                sample_entropies[method_key] = entropy_similarity
            except Exception as e:
                print(f"樣本熵計算錯誤 ({method_name}): {e}")
                sample_entropies[method_key] = 0.0
            
            # 計算綜合表現分數 (三種指標各佔1/3)
            try:
                # DTW分數：距離越小越好，轉換為相似性分數
                dtw_score = 1 / (1 + dtw_distances[method_key]) if dtw_distances[method_key] != float('inf') else 0
                
                # 頻譜相干性分數：直接使用 (0-1之間，越大越好)
                coherence_score = spectral_coherences[method_key]
                
                # 樣本熵相似性分數：直接使用 (0-1之間，越大越好)
                entropy_score = sample_entropies[method_key]
                
                # 綜合分數：三種指標各佔1/3
                composite_score = (dtw_score * (1/3) + coherence_score * (1/3) + 
                                 entropy_score * (1/3))
                composite_scores[method_key] = composite_score
                
            except Exception as e:
                print(f"綜合分數計算錯誤 ({method_name}): {e}")
                composite_scores[method_key] = 0.0
    
    # 以表格形式顯示結果
    print("\n自適應濾波方法性能比較表:")
    print("-" * 100)
    print("| {:^25} | {:^15} | {:^15} | {:^15} | {:^15} |".format(
        "濾波方法", "DTW距離", "頻譜相干性", "樣本熵相似性", "綜合表現分數"))
    print("|" + "-"*27 + "+" + "-"*17 + "+" + "-"*17 + "+" + "-"*17 + "+" + "-"*17 + "|")
    
    for method_key in ['1_raw', '2_lms', '3_rls', '4_nlms', '5_daf']:
        if method_key in results:
            method_name = method_names[method_key]
            dtw_val = dtw_distances.get(method_key, float('inf'))
            coherence_val = spectral_coherences.get(method_key, 0.0)
            entropy_val = sample_entropies.get(method_key, 0.0)
            composite_val = composite_scores.get(method_key, 0.0)
            
            print("| {:^25} | {:^15.4f} | {:^15.4f} | {:^15.4f} | {:^15.4f} |".format(
                method_name, dtw_val, coherence_val, entropy_val, composite_val))
    
    print("-" * 100)
    
    # 總結最佳方法
    print("\n各項指標最佳濾波方法總結:")
    print("-" * 60)
    
    # DTW距離越小越好
    if dtw_distances:
        best_dtw = min(dtw_distances.items(), key=lambda x: x[1])
        print(f"• 時間域指標 (DTW距離最小): {method_names[best_dtw[0]]:<25} ({best_dtw[1]:.4f})")
    
    # 頻譜相干性越大越好
    if spectral_coherences:
        best_coherence = max(spectral_coherences.items(), key=lambda x: x[1])
        print(f"• 頻域指標 (頻譜相干性最高): {method_names[best_coherence[0]]:<25} ({best_coherence[1]:.4f})")
    
    # 樣本熵相似性越大越好
    if sample_entropies:
        best_entropy = max(sample_entropies.items(), key=lambda x: x[1])
        print(f"• 非線性指標 (樣本熵最相似): {method_names[best_entropy[0]]:<25} ({best_entropy[1]:.4f})")
    
    # 綜合表現分數最高
    if composite_scores:
        best_composite = max(composite_scores.items(), key=lambda x: x[1])
        print(f"🏆 綜合表現分數最高: {method_names[best_composite[0]]:<25} ({best_composite[1]:.4f})")
        print(f"   (三種指標各佔1/3權重)")
    
    print("="*80)
    
    return {
        'dtw_distances': dtw_distances,
        'spectral_coherences': spectral_coherences,
        'sample_entropies': sample_entropies,
        'composite_scores': composite_scores,
        'method_names': method_names
    }

# %% [markdown]
# ## 6. 主函數

# %%
def main():
    """主程式入口 - 自適應濾波方法視覺化分析"""
    print("🚀 開始自適應濾波方法視覺化分析...")
    
    # 設定檔案路徑（請根據實際路徑修改）
    mmwave_file_path = r"C:\Users\jk121\Documents\Code\NEW_mmWave_PAPER\Output\merged_csv\45cm\05_20_2025_03_37_00.csv"
    ecg_file_path = r"C:\Users\jk121\Documents\Code\NEW_mmWave_PAPER\Output\ECG Data\CSV\45cm\2025-5-20, 337 AM-1.csv"
    
    print(f"\n{'='*80}")
    print("載入數據檔案")
    print(f"{'='*80}")
    print(f"mmWave 檔案: {mmwave_file_path}")
    print(f"ECG 檔案: {ecg_file_path}")
    
    # 檢查檔案是否存在
    if not os.path.exists(mmwave_file_path):
        print(f"❌ mmWave檔案不存在: {mmwave_file_path}")
        return None
    
    if not os.path.exists(ecg_file_path):
        print(f"❌ ECG檔案不存在: {ecg_file_path}")
        return None
    
    # 載入mmWave數據
    try:
        waveform_data, timestamps, heart_rates, frame_numbers, fs = load_and_preprocess_data(mmwave_file_path)
        print(f"✅ mmWave數據載入成功，數據長度: {len(waveform_data)} 樣本")
    except Exception as e:
        print(f"❌ mmWave數據載入失敗: {e}")
        return None
    
    # 載入ECG數據
    try:
        ecg_signal, ecg_time, ecg_fs = load_ecg_data(ecg_file_path)
        if ecg_signal is None or ecg_time is None:
            print(f"❌ ECG數據載入失敗")
            return None
        print(f"✅ ECG數據載入成功，數據長度: {len(ecg_signal)} 樣本")
    except Exception as e:
        print(f"❌ ECG數據載入失敗: {e}")
        return None
    
    # 進行自適應濾波處理
    print(f"\n{'='*80}")
    print("開始自適應濾波處理")
    print(f"{'='*80}")
    
    try:
        results = process_adaptive_filtering_methods(waveform_data, fs)
        print(f"✅ 自適應濾波處理完成，共產生 {len(results)} 種處理結果")
    except Exception as e:
        print(f"❌ 自適應濾波處理失敗: {e}")
        return None
    
    # 生成視覺化圖表
    print(f"\n{'='*80}")
    print("生成視覺化圖表")
    print(f"{'='*80}")
    
    try:
        mmwave_time = np.arange(len(results['1_raw'])) / fs
        output_filename = 'Adaptive_Filters_ECG_Comparison.png'
        visualize_adaptive_filtering_methods(results, fs, ecg_signal, ecg_time, output_filename)
        print("✅ 視覺化圖表生成完成")
    except Exception as e:
        print(f"❌ 視覺化圖表生成失敗: {e}")
        return None
    
    # 進行性能分析
    print(f"\n{'='*80}")
    print("開始性能分析")
    print(f"{'='*80}")
    
    try:
        mmwave_time = np.arange(len(results['1_raw'])) / fs
        analysis_results = analyze_adaptive_filtering_performance(results, ecg_signal, ecg_time, mmwave_time, fs)
        print("✅ 性能分析完成")
    except Exception as e:
        print(f"❌ 性能分析失敗: {e}")
        return None
    
    print(f"\n🎉 深度自適應濾波器視覺化分析完成!")
    print(f"📊 圖表已保存為: {output_filename}")
    print(f"📋 請查看上方的性能分析表格了解各濾波方法的效果")
    print(f"\n🔧 濾波器說明:")
    print(f"   • LMS: 最小均方自適應濾波器，收斂速度適中")
    print(f"   • RLS: 遞迴最小平方自適應濾波器，收斂速度快但計算複雜")
    print(f"   • NLMS: 正規化LMS自適應濾波器，穩定性較佳")
    print(f"   • DAF-CNN: 卷積神經網路深度自適應濾波器，善於特徵提取")
    print(f"   • DAF-LSTM: 長短期記憶網路深度自適應濾波器，適合時序建模")
    print(f"   • DAF-Hybrid: CNN+LSTM混合深度自適應濾波器，綜合優勢最佳")
    print(f"   • DAF-Transformer: 基於注意力機制的深度自適應濾波器 (如可用)")
    
    return results, analysis_results

# %%
if __name__ == "__main__":
    results, analysis = main()
