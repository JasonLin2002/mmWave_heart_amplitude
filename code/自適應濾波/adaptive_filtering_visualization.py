# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from scipy import linalg
import os
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from dtaidistance import dtw
from scipy.signal import coherence
from scipy.stats import entropy
import tensorflow as tf
from tensorflow import keras

# %% [markdown]
# ## 1. 數據載入和預處理函數

# %%
def load_ecg_data(ecg_path):
    """Load and preprocess ECG data (last 60 seconds)."""
    print(f"Loading ECG data from: {ecg_path}")
    
    try:
        ecg_df = pd.read_csv(ecg_path)
        print(f"ECG data shape: {ecg_df.shape}")
        print("\nECG data columns:")
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
        
        print(f"ECG signal length: {len(ecg_signal)} samples ({len(ecg_signal)/ecg_fs:.1f} seconds)")
        print(f"ECG time range: {ecg_time[0]:.1f} to {ecg_time[-1]:.1f} seconds")
        
        return ecg_signal, ecg_time, ecg_fs
        
    except Exception as e:
        print(f"Error loading ECG data: {e}")
        return None, None, None

def load_and_preprocess_data(file_path):
    """Read and preprocess radar heart amplitude data (only last 60 seconds)."""
    print(f"Reading file: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    print("\nData preview:")
    print(df.head())
    fs = 11.11  # Fixed sampling rate
    print(f"\nEstimated sampling rate: {fs:.2f} Hz")
    df_sorted = df.sort_values(by='Frame_Number')
    waveform_data = df_sorted['Heart_Waveform'].values
    timestamps = df_sorted['Timestamp'].values
    heart_rates = df_sorted['Heart_Rate'].values
    frame_numbers = df_sorted['Frame_Number'].values

    # Only keep last 60 seconds
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
# ## 2. 自適應濾波器實現

# %%
class LMSFilter:
    """最小均方 (LMS) 自適應濾波器"""
    
    def __init__(self, n_taps=32, mu=0.01):
        self.n_taps = n_taps
        self.mu = mu
        self.weights = np.zeros(n_taps)
        self.buffer = np.zeros(n_taps)
        
    def filter(self, input_signal, desired_signal):
        """應用LMS濾波"""
        output = []
        error_history = []
        
        for i in range(len(input_signal)):
            # 更新輸入緩衝區
            self.buffer = np.roll(self.buffer, -1)
            self.buffer[-1] = input_signal[i]
            
            # 計算輸出
            y = np.dot(self.weights, self.buffer)
            output.append(y)
            
            # 計算誤差
            if i < len(desired_signal):
                error = desired_signal[i] - y
                error_history.append(error)
                
                # 更新權重
                self.weights += self.mu * error * self.buffer
            else:
                error_history.append(0)
        
        return np.array(output), np.array(error_history)

class RLSFilter:
    """遞歸最小二乘 (RLS) 自適應濾波器"""
    
    def __init__(self, n_taps=32, forgetting_factor=0.99, reg_param=1e-4):
        self.n_taps = n_taps
        self.lam = forgetting_factor
        self.weights = np.zeros(n_taps)
        self.P = np.eye(n_taps) / reg_param
        self.buffer = np.zeros(n_taps)
        
    def filter(self, input_signal, desired_signal):
        """應用RLS濾波"""
        output = []
        error_history = []
        
        for i in range(len(input_signal)):
            # 更新輸入緩衝區
            self.buffer = np.roll(self.buffer, -1)
            self.buffer[-1] = input_signal[i]
            
            # 計算輸出
            y = np.dot(self.weights, self.buffer)
            output.append(y)
            
            # 計算誤差並更新權重
            if i < len(desired_signal):
                error = desired_signal[i] - y
                error_history.append(error)
                
                # RLS權重更新
                k = (self.P @ self.buffer) / (self.lam + self.buffer.T @ self.P @ self.buffer)
                self.P = (self.P - np.outer(k, self.buffer.T @ self.P)) / self.lam
                self.weights += k * error
            else:
                error_history.append(0)
                
        return np.array(output), np.array(error_history)

class NLMSFilter:
    """正規化LMS (NLMS) 自適應濾波器"""
    
    def __init__(self, n_taps=32, mu=0.5, eps=1e-8):
        self.n_taps = n_taps
        self.mu = mu
        self.eps = eps
        self.weights = np.zeros(n_taps)
        self.buffer = np.zeros(n_taps)
        
    def filter(self, input_signal, desired_signal):
        """應用NLMS濾波"""
        output = []
        error_history = []
        
        for i in range(len(input_signal)):
            # 更新輸入緩衝區
            self.buffer = np.roll(self.buffer, -1)
            self.buffer[-1] = input_signal[i]
            
            # 計算輸出
            y = np.dot(self.weights, self.buffer)
            output.append(y)
            
            # 計算誤差
            if i < len(desired_signal):
                error = desired_signal[i] - y
                error_history.append(error)
                
                # NLMS權重更新 (正規化步長)
                norm_factor = np.dot(self.buffer, self.buffer) + self.eps
                self.weights += (self.mu * error / norm_factor) * self.buffer
            else:
                error_history.append(0)
                
        return np.array(output), np.array(error_history)

class DeepAdaptiveFilter:
    """深度自適應濾波器 (DAF)"""
    
    def __init__(self, input_dim=64, model_path=None, model_type='cnn'):
        self.input_dim = input_dim
        self.model_type = model_type
        self.model = None
        
        if model_path and os.path.exists(model_path):
            try:
                # 嘗試多種載入方式來解決序列化問題
                print(f"🔄 正在載入 {model_type.upper()} 模型: {model_path}")
                
                # 方法1：使用 compile=False 來避免編譯相關的序列化問題
                try:
                    self.model = tf.keras.models.load_model(model_path, compile=False)
                    # 手動重新編譯模型
                    self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    print(f"✅ 成功載入您訓練的 {model_type.upper()} 模型 (方法1)")
                except Exception as e1:
                    print(f"🔄 方法1失敗: {e1}")
                    
                    # 方法2：使用 custom_objects 來註冊自定義函數
                    try:
                        custom_objects = {
                            'mse': tf.keras.losses.MeanSquaredError(),
                            'mae': tf.keras.metrics.MeanAbsoluteError(),
                            'mean_squared_error': tf.keras.losses.MeanSquaredError(),
                            'mean_absolute_error': tf.keras.metrics.MeanAbsoluteError()
                        }
                        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                        print(f"✅ 成功載入您訓練的 {model_type.upper()} 模型 (方法2)")
                    except Exception as e2:
                        print(f"🔄 方法2失敗: {e2}")
                        
                        # 方法3：載入權重而非完整模型
                        try:
                            self.model = self._create_architecture_model()
                            self.model.load_weights(model_path.replace('.h5', '_weights.h5'))
                            print(f"✅ 成功載入您訓練的 {model_type.upper()} 模型權重 (方法3)")
                        except Exception as e3:
                            print(f"🔄 方法3失敗: {e3}")
                            print(f"⚠️ 所有載入方法都失敗，使用備用模型")
                            self.model = self._create_fallback_model()
                            
            except Exception as e:
                print(f"❌ 載入 {model_type.upper()} 模型時發生意外錯誤: {e}")
                self.model = self._create_fallback_model()
        else:
            print(f"⚠️ {model_type.upper()} 模型檔案不存在: {model_path}")
            self.model = self._create_fallback_model()
    
    def _create_architecture_model(self):
        """創建與訓練模型相同架構的模型"""
        try:
            if self.model_type == 'cnn':
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(self.input_dim, 1)),
                    tf.keras.layers.Conv1D(64, 3, activation='relu'),
                    tf.keras.layers.GlobalMaxPooling1D(),
                    tf.keras.layers.Dense(50, activation='relu'),
                    tf.keras.layers.Dense(1, activation='linear')
                ])
            elif self.model_type == 'lstm':
                model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(50, input_shape=(self.input_dim, 1), return_sequences=True),
                    tf.keras.layers.LSTM(50),
                    tf.keras.layers.Dense(25, activation='relu'),
                    tf.keras.layers.Dense(1, activation='linear')
                ])
            else:  # hybrid
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(self.input_dim, 1)),
                    tf.keras.layers.LSTM(50),
                    tf.keras.layers.Dense(25, activation='relu'),
                    tf.keras.layers.Dense(1, activation='linear')
                ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        except Exception as e:
            print(f"❌ 創建架構模型失敗: {e}")
            return None

    def _create_fallback_model(self):
        """創建備用模型（當所有載入方法都失敗時使用）"""
        try:
            model = self._create_architecture_model()
            if model is not None:
                print(f"⚠️ 使用未訓練的備用 {self.model_type.upper()} 模型")
            return model
        except Exception as e:
            print(f"❌ 創建備用模型失敗: {e}")
            return None
    
    def filter(self, input_signal):
        """應用深度自適應濾波"""
        if self.model is None:
            print(f"❌ {self.model_type.upper()} 模型未載入，返回原始信號")
            return input_signal
        
        try:
            # 標準化輸入信號
            scaler = StandardScaler()
            signal_scaled = scaler.fit_transform(input_signal.reshape(-1, 1)).flatten()
            
            # 創建滑動窗口
            windows = []
            for i in range(len(signal_scaled) - self.input_dim + 1):
                windows.append(signal_scaled[i:i + self.input_dim])
            
            if len(windows) == 0:
                print(f"⚠️ 信號長度不足，無法創建窗口")
                return input_signal
                
            X = np.array(windows).reshape(len(windows), self.input_dim, 1)
            
            # 預測
            predictions = self.model.predict(X, verbose=0)
            
            # 重建完整信號長度
            output = np.zeros(len(input_signal))
            output[:self.input_dim-1] = signal_scaled[:self.input_dim-1]  # 填充前面部分
            output[self.input_dim-1:] = predictions.flatten()
            
            # 反標準化
            output_rescaled = scaler.inverse_transform(output.reshape(-1, 1)).flatten()
            
            print(f"✅ {self.model_type.upper()} 濾波完成，處理 {len(input_signal)} 個樣本")
            return output_rescaled
            
        except Exception as e:
            print(f"❌ {self.model_type.upper()} 濾波處理失敗: {e}")
            return input_signal

# %% [markdown]
# ## 3. 自適應濾波處理函數

# %%
def process_adaptive_filtering_methods(waveform_data, fs, distance='45cm'):
    """
    自適應濾波方法處理
    
    處理流程：
    1. 原始訊號
    2. LMS 自適應濾波
    3. RLS 自適應濾波  
    4. NLMS 自適應濾波
    5. DAF-CNN 深度自適應濾波器
    6. DAF-LSTM 深度自適應濾波器
    7. DAF-Hybrid 深度自適應濾波器
    """
    
    results = {}
    
    print("🔄 開始自適應濾波處理...")
    
    # 1. 原始訊號
    results['1_raw'] = waveform_data
    print("✅ 1. 原始訊號")
    
    # 前處理：去趨勢和標準化
    detrended = signal.detrend(waveform_data)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(detrended.reshape(-1, 1)).flatten()
    
    # 創建期望信號（使用輕微的低通濾波作為參考）
    try:
        b, a = signal.butter(4, 0.3, btype='lowpass')
        desired_signal = signal.filtfilt(b, a, normalized)
    except:
        desired_signal = normalized
    
    # 2. LMS 自適應濾波
    try:
        lms_filter = LMSFilter(n_taps=32, mu=0.01)
        lms_output, lms_error = lms_filter.filter(normalized, desired_signal)
        results['2_lms'] = scaler.inverse_transform(lms_output.reshape(-1, 1)).flatten()
        print("✅ 2. LMS 自適應濾波")
    except Exception as e:
        print(f"❌ LMS 濾波失敗: {e}")
        results['2_lms'] = detrended
    
    # 3. RLS 自適應濾波
    try:
        rls_filter = RLSFilter(n_taps=32, forgetting_factor=0.99)
        rls_output, rls_error = rls_filter.filter(normalized, desired_signal)
        results['3_rls'] = scaler.inverse_transform(rls_output.reshape(-1, 1)).flatten()
        print("✅ 3. RLS 自適應濾波")
    except Exception as e:
        print(f"❌ RLS 濾波失敗: {e}")
        results['3_rls'] = detrended
    
    # 4. NLMS 自適應濾波
    try:
        nlms_filter = NLMSFilter(n_taps=32, mu=0.5)
        nlms_output, nlms_error = nlms_filter.filter(normalized, desired_signal)
        results['4_nlms'] = scaler.inverse_transform(nlms_output.reshape(-1, 1)).flatten()
        print("✅ 4. NLMS 自適應濾波")
    except Exception as e:
        print(f"❌ NLMS 濾波失敗: {e}")
        results['4_nlms'] = detrended
    
    # 5. DAF-CNN 深度自適應濾波器
    cnn_model_path = f"mmWave_heart_amplitude/code/trained_models/daf_cnn_{distance}.h5"
    try:
        daf_cnn_filter = DeepAdaptiveFilter(input_dim=64, model_path=cnn_model_path, model_type='cnn')
        cnn_output = daf_cnn_filter.filter(detrended)
        results['5_daf_cnn'] = cnn_output
        print("✅ 5. DAF-CNN 深度自適應濾波器")
    except Exception as e:
        print(f"❌ DAF-CNN 濾波失敗: {e}")
        results['5_daf_cnn'] = detrended
    
    # 6. DAF-LSTM 深度自適應濾波器
    lstm_model_path = f"mmWave_heart_amplitude/code/trained_models/daf_lstm_{distance}.h5"
    try:
        daf_lstm_filter = DeepAdaptiveFilter(input_dim=64, model_path=lstm_model_path, model_type='lstm')
        lstm_output = daf_lstm_filter.filter(detrended)
        results['6_daf_lstm'] = lstm_output
        print("✅ 6. DAF-LSTM 深度自適應濾波器")
    except Exception as e:
        print(f"❌ DAF-LSTM 濾波失敗: {e}")
        results['6_daf_lstm'] = detrended
    
    # 7. DAF-Hybrid 深度自適應濾波器
    hybrid_model_path = f"mmWave_heart_amplitude/code/trained_models/daf_hybrid_{distance}.h5"
    try:
        daf_hybrid_filter = DeepAdaptiveFilter(input_dim=64, model_path=hybrid_model_path, model_type='hybrid')
        hybrid_output = daf_hybrid_filter.filter(detrended)
        results['7_daf_hybrid'] = hybrid_output
        print("✅ 7. DAF-Hybrid 深度自適應濾波器")
    except Exception as e:
        print(f"❌ DAF-Hybrid 濾波失敗: {e}")
        results['7_daf_hybrid'] = detrended
    
    print("🎉 所有自適應濾波處理完成!")
    
    return results

# %% [markdown]
# ## 4. 視覺化函數

# %%
def visualize_adaptive_filtering_methods(results, fs, ecg_signal=None, ecg_time=None, output_filename=None):
    """可視化自適應濾波方法與ECG的比較 (7個子圖)"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互模式
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"Font setting error: {e}")

    time = np.arange(len(results['1_raw'])) / fs
    
    # 定義7種方法的詳細名稱和顏色
    methods = [
        ('1_raw', '原始訊號 (Raw Signal)', 'blue'),
        ('2_lms', 'LMS 自適應濾波器', 'red'),
        ('3_rls', 'RLS 自適應濾波器', 'green'),
        ('4_nlms', 'NLMS 自適應濾波器', 'orange'),
        ('5_daf_cnn', 'DAF-CNN 深度濾波器', 'purple'),
        ('6_daf_lstm', 'DAF-LSTM 深度濾波器', 'brown'),
        ('7_daf_hybrid', 'DAF-Hybrid 深度濾波器', 'pink')
    ]

    plt.figure(figsize=(20, 28))  # 增大圖片尺寸
    
    for i, (method_key, method_name, color) in enumerate(methods):
        if method_key in results:
            ax = plt.subplot(4, 2, i+1)
            ax.plot(time, results[method_key], color=color, linewidth=1.0, label=f'mmWave {method_name}')
            ax.set_title(f'{method_name} vs ECG', fontsize=12, fontweight='bold')
            ax.set_ylabel('mmWave Amplitude', color=color, fontsize=10)
            ax.tick_params(axis='y', labelcolor=color)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 60)
            
            # 添加ECG參考線
            if ecg_signal is not None and ecg_time is not None:
                ax_twin = ax.twinx()
                ax_twin.plot(ecg_time, ecg_signal, 'k-', alpha=0.6, linewidth=0.8, label='ECG Reference')
                ax_twin.set_ylabel('ECG Amplitude', color='k', fontsize=10)
                ax_twin.tick_params(axis='y', labelcolor='k')
                
                # 添加圖例
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax_twin.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
            else:
                ax.legend(loc='upper right', fontsize=8)
            
            # 設置x軸刻度
            ax.set_xticks(np.arange(0, 61, 10))
            ax.set_xticklabels([str(x) for x in range(0, 61, 10)])
            
            if i >= 5:  # 最後一行添加x軸標籤
                ax.set_xlabel('Time (seconds)', fontsize=10)

    plt.tight_layout()
    
    # 保存圖片
    if output_filename is None:
        output_filename = 'Adaptive_Filtering_Methods_ECG_Comparison.png'
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"📊 自適應濾波方法比較圖已儲存為: {output_filename}")
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
    """分析自適應濾波方法的性能"""
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
        '5_daf_cnn': '5.DAF-CNN深度濾波器',
        '6_daf_lstm': '6.DAF-LSTM深度濾波器',
        '7_daf_hybrid': '7.DAF-Hybrid深度濾波器'
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
    print("| {:^30} | {:^15} | {:^15} | {:^15} | {:^15} |".format(
        "濾波方法", "DTW距離", "頻譜相干性", "樣本熵相似性", "綜合表現分數"))
    print("|" + "-"*32 + "+" + "-"*17 + "+" + "-"*17 + "+" + "-"*17 + "+" + "-"*17 + "|")
    
    for method_key in ['1_raw', '2_lms', '3_rls', '4_nlms', '5_daf_cnn', '6_daf_lstm', '7_daf_hybrid']:
        if method_key in results:
            method_name = method_names[method_key]
            dtw_val = dtw_distances.get(method_key, float('inf'))
            coherence_val = spectral_coherences.get(method_key, 0.0)
            entropy_val = sample_entropies.get(method_key, 0.0)
            composite_val = composite_scores.get(method_key, 0.0)
            
            print("| {:^30} | {:^15.4f} | {:^15.4f} | {:^15.4f} | {:^15.4f} |".format(
                method_name, dtw_val, coherence_val, entropy_val, composite_val))
    
    print("-" * 100)
    
    # 總結最佳方法
    print("\n各項指標最佳濾波方法總結:")
    print("-" * 70)
    
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
    
    # 深度學習方法排名
    deep_methods = ['5_daf_cnn', '6_daf_lstm', '7_daf_hybrid']
    deep_scores = {k: v for k, v in composite_scores.items() if k in deep_methods}
    if deep_scores:
        print(f"\n🤖 深度學習方法排名:")
        sorted_deep = sorted(deep_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (method_key, score) in enumerate(sorted_deep, 1):
            status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"   {status} {method_names[method_key]}: {score:.4f}")
    
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
    mmwave_file_path = r"NEW_mmWave_PAPER/Output/merged_csv/45cm/05_20_2025_03_37_00.csv"
    ecg_file_path = r"NEW_mmWave_PAPER/Output/ECG Data/CSV/45cm/2025-5-20, 337 AM-1.csv"
    distance = '45cm'  # 可以修改為其他距離: 30cm, 60cm, 90cm
    
    print(f"\n{'='*80}")
    print("載入數據檔案")
    print(f"{'='*80}")
    print(f"mmWave 檔案: {mmwave_file_path}")
    print(f"ECG 檔案: {ecg_file_path}")
    print(f"距離: {distance}")
    
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
        results = process_adaptive_filtering_methods(waveform_data, fs, distance)
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
        output_filename = f'Adaptive_Filtering_Methods_{distance}_ECG_Comparison.png'
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
    
    print(f"\n🎉 自適應濾波方法視覺化分析完成!")
    print(f"📊 圖表已保存為: {output_filename}")
    print(f"📋 請查看上方的性能分析表格了解各濾波方法的效果")
    
    return results, analysis_results

# %%
if __name__ == "__main__":
    results, analysis = main()
