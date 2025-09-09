# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
import pickle
import warnings
warnings.filterwarnings('ignore')

# 設定 TensorFlow GPU 使用
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"找到 {len(gpus)} 個 GPU，已設定動態記憶體分配")
    except RuntimeError as e:
        print(f"GPU 設定錯誤: {e}")
else:
    print("未找到 GPU，將使用 CPU 進行訓練")

# %% [markdown]
# ## 1. 數據載入和預處理類別

# %%
class DataLoader:
    """數據載入和預處理類別"""
    
    def __init__(self, mmwave_base_path, ecg_base_path):
        self.mmwave_base_path = mmwave_base_path
        self.ecg_base_path = ecg_base_path
        self.distances = ['30cm', '45cm', '60cm', '90cm']
        
    def load_ecg_data(self, ecg_file_path):
        """載入單個ECG檔案"""
        try:
            ecg_df = pd.read_csv(ecg_file_path)
            
            if 'time' not in ecg_df.columns or 'ecg' not in ecg_df.columns:
                print(f"警告: {ecg_file_path} 缺少必要欄位")
                return None, None
                
            df_cleaned = ecg_df.dropna(subset=['ecg'])
            ecg_signal = df_cleaned['ecg'].values
            timestamps_ns = df_cleaned['time'].values
            
            if len(timestamps_ns) < 2:
                return None, None
                
            # 轉換為相對時間（秒）
            times_s = timestamps_ns * 1e-9
            ecg_time = times_s - times_s[0]
            
            return ecg_signal, ecg_time
            
        except Exception as e:
            print(f"載入ECG數據錯誤: {ecg_file_path}, {e}")
            return None, None
    
    def load_mmwave_data(self, mmwave_file_path):
        """載入單個mmWave檔案"""
        try:
            df = pd.read_csv(mmwave_file_path)
            
            if 'Heart_Waveform' not in df.columns:
                print(f"警告: {mmwave_file_path} 缺少 Heart_Waveform 欄位")
                return None, None
                
            df_sorted = df.sort_values(by='Frame_Number')
            waveform_data = df_sorted['Heart_Waveform'].values
            
            # 創建時間軸 (假設固定採樣率 11.11 Hz)
            fs = 11.11
            time_axis = np.arange(len(waveform_data)) / fs
            
            return waveform_data, time_axis
            
        except Exception as e:
            print(f"載入mmWave數據錯誤: {mmwave_file_path}, {e}")
            return None, None
    
    def resample_signals(self, mmwave_signal, mmwave_time, ecg_signal, ecg_time, target_length=None):
        """重採樣信號到相同長度"""
        try:
            # 確定目標長度
            if target_length is None:
                target_length = min(len(mmwave_signal), len(ecg_signal))
            
            # 重採樣到統一時間軸
            if len(mmwave_time) > 1 and len(ecg_time) > 1:
                # 創建統一時間軸
                max_time = min(mmwave_time[-1], ecg_time[-1])
                unified_time = np.linspace(0, max_time, target_length)
                
                # 重採樣信號
                mmwave_resampled = np.interp(unified_time, mmwave_time, mmwave_signal)
                ecg_resampled = np.interp(unified_time, ecg_time, ecg_signal)
                
                return mmwave_resampled, ecg_resampled, unified_time
            else:
                return None, None, None
                
        except Exception as e:
            print(f"重採樣錯誤: {e}")
            return None, None, None
    
    def create_training_pairs(self, distance='45cm', max_pairs_per_distance=50):
        """創建訓練數據對"""
        mmwave_folder = os.path.join(self.mmwave_base_path, distance)
        ecg_folder = os.path.join(self.ecg_base_path, distance)
        
        if not os.path.exists(mmwave_folder) or not os.path.exists(ecg_folder):
            print(f"資料夾不存在: {mmwave_folder} 或 {ecg_folder}")
            return [], [], []
        
        # 取得檔案列表
        mmwave_files = sorted(glob.glob(os.path.join(mmwave_folder, "*.csv")))
        ecg_files = sorted(glob.glob(os.path.join(ecg_folder, "*.csv")))
        
        print(f"找到 mmWave 檔案: {len(mmwave_files)} 個")
        print(f"找到 ECG 檔案: {len(ecg_files)} 個")
        
        mmwave_data = []
        ecg_data = []
        timestamps = []
        
        # 限制處理的檔案數量以避免記憶體問題
        max_files = min(len(mmwave_files), len(ecg_files), max_pairs_per_distance)
        
        for i in range(max_files):
            mmwave_signal, mmwave_time = self.load_mmwave_data(mmwave_files[i])
            ecg_signal, ecg_time = self.load_ecg_data(ecg_files[i])
            
            if mmwave_signal is not None and ecg_signal is not None:
                # 重採樣到相同長度
                mmwave_resampled, ecg_resampled, unified_time = self.resample_signals(
                    mmwave_signal, mmwave_time, ecg_signal, ecg_time, target_length=600  # 約60秒的數據
                )
                
                if mmwave_resampled is not None and ecg_resampled is not None:
                    mmwave_data.append(mmwave_resampled)
                    ecg_data.append(ecg_resampled)
                    timestamps.append(unified_time)
                    
                    if len(mmwave_data) % 10 == 0:
                        print(f"已處理 {len(mmwave_data)} 對數據...")
        
        print(f"成功創建 {len(mmwave_data)} 對訓練數據 ({distance})")
        return mmwave_data, ecg_data, timestamps

# %% [markdown]
# ## 2. 深度自適應濾波器模型

# %%
class DeepAdaptiveFilterModel:
    """深度自適應濾波器模型"""
    
    def __init__(self, input_window_size=64, model_type='hybrid'):
        self.input_window_size = input_window_size
        self.model_type = model_type  # 'cnn', 'lstm', 'hybrid', 'transformer'
        self.model = None
        self.scaler_input = StandardScaler()
        self.scaler_output = StandardScaler()
        self.is_fitted = False
        
    def build_cnn_model(self):
        """建構CNN模型"""
        model = Sequential([
            Input(shape=(self.input_window_size, 1)),
            Conv1D(64, kernel_size=5, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        return model
    
    def build_lstm_model(self):
        """建構LSTM模型"""
        model = Sequential([
            Input(shape=(self.input_window_size, 1)),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        return model
    
    def build_hybrid_model(self):
        """建構混合CNN-LSTM模型"""
        input_layer = Input(shape=(self.input_window_size, 1))
        
        # CNN 分支
        cnn_branch = Conv1D(64, kernel_size=5, activation='relu', padding='same')(input_layer)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        cnn_branch = Conv1D(32, kernel_size=3, activation='relu', padding='same')(cnn_branch)
        
        # LSTM 分支
        lstm_branch = LSTM(64, return_sequences=True)(input_layer)
        lstm_branch = LSTM(32, return_sequences=False)(lstm_branch)
        
        # 合併分支
        merged = layers.concatenate([Flatten()(cnn_branch), lstm_branch])
        
        # 全連接層
        dense_layer = Dense(128, activation='relu')(merged)
        dense_layer = Dropout(0.3)(dense_layer)
        dense_layer = Dense(64, activation='relu')(dense_layer)
        dense_layer = Dropout(0.2)(dense_layer)
        output_layer = Dense(1, activation='linear')(dense_layer)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        return model
    
    def build_transformer_model(self):
        """建構Transformer模型"""
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Multi-head self-attention
            attention_layer = layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )
            attention_output = attention_layer(inputs, inputs)
            attention_output = layers.Dropout(dropout)(attention_output)
            attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
            
            # Feed-forward network
            ffn = Sequential([
                Dense(ff_dim, activation='relu'),
                Dense(inputs.shape[-1])
            ])
            ffn_output = ffn(attention_output)
            ffn_output = layers.Dropout(dropout)(ffn_output)
            return layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        
        inputs = Input(shape=(self.input_window_size, 1))
        
        # 位置編碼
        x = layers.Dense(64)(inputs)
        
        # Transformer 編碼器
        x = transformer_encoder(x, head_size=16, num_heads=4, ff_dim=128, dropout=0.1)
        x = transformer_encoder(x, head_size=16, num_heads=4, ff_dim=128, dropout=0.1)
        
        # 全域平均池化和輸出
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = Model(inputs, outputs)
        return model
    
    def build_model(self):
        """根據模型類型建構模型"""
        if self.model_type == 'cnn':
            model = self.build_cnn_model()
        elif self.model_type == 'lstm':
            model = self.build_lstm_model()
        elif self.model_type == 'hybrid':
            model = self.build_hybrid_model()
        elif self.model_type == 'transformer':
            model = self.build_transformer_model()
        else:
            raise ValueError(f"不支援的模型類型: {self.model_type}")
        
        # 編譯模型
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_training_data(self, mmwave_signals, ecg_signals):
        """準備訓練數據"""
        X_sequences = []
        y_sequences = []
        
        print("正在準備訓練數據...")
        
        for i, (mmwave_signal, ecg_signal) in enumerate(zip(mmwave_signals, ecg_signals)):
            # 正規化信號
            mmwave_norm = (mmwave_signal - np.mean(mmwave_signal)) / np.std(mmwave_signal)
            ecg_norm = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
            
            # 創建滑動窗口
            for j in range(len(mmwave_norm) - self.input_window_size):
                X_sequences.append(mmwave_norm[j:j + self.input_window_size])
                y_sequences.append(ecg_norm[j + self.input_window_size])
            
            if (i + 1) % 10 == 0:
                print(f"已處理 {i + 1}/{len(mmwave_signals)} 個信號...")
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        # 重塑為 (samples, time_steps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        print(f"訓練數據形狀: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def train(self, mmwave_signals, ecg_signals, validation_split=0.2, epochs=100, batch_size=32, save_path=None):
        """訓練模型"""
        print(f"開始訓練深度自適應濾波器 ({self.model_type} 模型)...")
        
        # 準備訓練數據
        X, y = self.prepare_training_data(mmwave_signals, ecg_signals)
        
        # 分割訓練和驗證數據
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        print(f"訓練集大小: {X_train.shape[0]} 樣本")
        print(f"驗證集大小: {X_val.shape[0]} 樣本")
        
        # 建構模型
        self.model = self.build_model()
        
        # 顯示模型架構
        print("\n模型架構:")
        self.model.summary()
        
        # 設定回調函數
        callback_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
        ]
        
        if save_path:
            checkpoint_callback = callbacks.ModelCheckpoint(
                filepath=save_path + '_checkpoint.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
            callback_list.append(checkpoint_callback)
        
        # 訓練模型
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.is_fitted = True
        
        # 保存模型和訓練歷史
        if save_path:
            self.model.save(save_path + '.h5')
            
            # 保存訓練歷史
            with open(save_path + '_history.pkl', 'wb') as f:
                pickle.dump(history.history, f)
            
            print(f"模型已保存到: {save_path}.h5")
            print(f"訓練歷史已保存到: {save_path}_history.pkl")
        
        return history
    
    def filter_signal(self, input_signal):
        """使用訓練好的模型濾波信號"""
        if not self.is_fitted or self.model is None:
            raise ValueError("模型尚未訓練或載入")
        
        # 正規化輸入信號
        input_norm = (input_signal - np.mean(input_signal)) / np.std(input_signal)
        
        # 準備預測數據
        X_pred = []
        for i in range(len(input_norm) - self.input_window_size):
            X_pred.append(input_norm[i:i + self.input_window_size])
        
        X_pred = np.array(X_pred).reshape(-1, self.input_window_size, 1)
        
        # 預測
        predictions = self.model.predict(X_pred, verbose=0)
        
        # 組合輸出信號
        output_signal = np.zeros(len(input_signal))
        output_signal[:self.input_window_size] = input_norm[:self.input_window_size]
        output_signal[self.input_window_size:] = predictions.flatten()
        
        return output_signal

# %% [markdown]
# ## 3. 訓練管理器

# %%
class TrainingManager:
    """訓練管理器"""
    
    def __init__(self, mmwave_base_path, ecg_base_path):
        self.data_loader = DataLoader(mmwave_base_path, ecg_base_path)
        self.models = {}
        
    def train_all_models(self, distances=['45cm'], model_types=['hybrid'], 
                        max_pairs_per_distance=30, epochs=50, save_dir='models'):
        """訓練所有模型組合"""
        
        # 創建模型保存目錄
        os.makedirs(save_dir, exist_ok=True)
        
        results = {}
        
        for distance in distances:
            print(f"\n{'='*80}")
            print(f"處理距離: {distance}")
            print(f"{'='*80}")
            
            # 載入數據
            mmwave_data, ecg_data, timestamps = self.data_loader.create_training_pairs(
                distance=distance, 
                max_pairs_per_distance=max_pairs_per_distance
            )
            
            if len(mmwave_data) == 0:
                print(f"警告: {distance} 沒有可用的訓練數據")
                continue
            
            for model_type in model_types:
                print(f"\n訓練 {model_type} 模型 ({distance})...")
                
                # 創建模型
                model = DeepAdaptiveFilterModel(
                    input_window_size=64, 
                    model_type=model_type
                )
                
                # 設定保存路徑
                save_path = os.path.join(save_dir, f'daf_{model_type}_{distance.replace("cm", "")}cm')
                
                try:
                    # 訓練模型
                    history = model.train(
                        mmwave_signals=mmwave_data,
                        ecg_signals=ecg_data,
                        validation_split=0.2,
                        epochs=epochs,
                        batch_size=32,
                        save_path=save_path
                    )
                    
                    # 保存模型實例
                    model_key = f"{distance}_{model_type}"
                    self.models[model_key] = model
                    results[model_key] = {
                        'model': model,
                        'history': history.history,
                        'save_path': save_path + '.h5'
                    }
                    
                    print(f"✅ {model_type} 模型 ({distance}) 訓練完成")
                    
                except Exception as e:
                    print(f"❌ {model_type} 模型 ({distance}) 訓練失敗: {e}")
                    continue
        
        return results
    
    def plot_training_history(self, results, save_path='training_history.png'):
        """繪製訓練歷史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('深度自適應濾波器訓練歷史', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_key, result) in enumerate(results.items()):
            history = result['history']
            color = colors[i % len(colors)]
            
            # Loss
            axes[0, 0].plot(history['loss'], color=color, label=f'{model_key} - 訓練')
            axes[0, 0].plot(history['val_loss'], color=color, linestyle='--', label=f'{model_key} - 驗證')
            axes[0, 0].set_title('訓練損失')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss (MSE)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # MAE
            axes[0, 1].plot(history['mae'], color=color, label=f'{model_key} - 訓練')
            axes[0, 1].plot(history['val_mae'], color=color, linestyle='--', label=f'{model_key} - 驗證')
            axes[0, 1].set_title('平均絕對誤差')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 隱藏未使用的子圖
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"訓練歷史圖表已保存: {save_path}")
        plt.show()
        plt.close()

# %% [markdown]
# ## 4. 主程式

# %%
def main():
    """主程式入口"""
    print("🚀 深度自適應濾波器訓練程式")
    print("="*80)
    
    # 設定路徑
    mmwave_base_path = r"C:\Users\jk121\Documents\Code\mmWave-PAPER\Output\merged_csv"
    ecg_base_path = r"C:\Users\jk121\Documents\Code\mmWave-PAPER\Output\ECG Data\CSV"
    
    print(f"mmWave 數據路徑: {mmwave_base_path}")
    print(f"ECG 數據路徑: {ecg_base_path}")
    
    # 檢查路徑是否存在
    if not os.path.exists(mmwave_base_path):
        print(f"❌ mmWave 數據路徑不存在: {mmwave_base_path}")
        return
    
    if not os.path.exists(ecg_base_path):
        print(f"❌ ECG 數據路徑不存在: {ecg_base_path}")
        return
    
    # 創建訓練管理器
    trainer = TrainingManager(mmwave_base_path, ecg_base_path)
    
    # 設定訓練參數
    distances = ['30cm', '45cm', '60cm', '90cm']  # 可以選擇特定距離
    model_types = ['hybrid', 'cnn', 'lstm']       # 可以選擇特定模型類型
    max_pairs_per_distance = 20                    # 每個距離最多使用的檔案數
    epochs = 50                                    # 訓練輪數
    
    print(f"\n訓練設定:")
    print(f"  距離: {distances}")
    print(f"  模型類型: {model_types}")
    print(f"  每距離最大檔案數: {max_pairs_per_distance}")
    print(f"  訓練輪數: {epochs}")
    
    # 開始訓練
    print(f"\n開始批次訓練...")
    results = trainer.train_all_models(
        distances=distances,
        model_types=model_types,
        max_pairs_per_distance=max_pairs_per_distance,
        epochs=epochs,
        save_dir='mmWave_heart_amplitude/code/trained_models'
    )
    
    # 繪製訓練歷史
    if results:
        trainer.plot_training_history(results, 'mmWave_heart_amplitude/code/training_history.png')
        
        print(f"\n🎉 訓練完成!")
        print(f"訓練結果總結:")
        for model_key, result in results.items():
            print(f"  ✅ {model_key}: {result['save_path']}")
            
        print(f"\n📋 使用說明:")
        print(f"  1. 訓練好的模型已保存在 'mmWave_heart_amplitude/code/trained_models/' 目錄")
        print(f"  2. 可以在 adaptive_filters_comparison.py 中載入這些模型")
        print(f"  3. 訓練歷史圖表: mmWave_heart_amplitude/code/training_history.png")
        
    else:
        print("❌ 沒有成功訓練任何模型")

# 快速訓練函數（僅針對特定配置）
def quick_train():
    """快速訓練函數 - 僅訓練45cm的hybrid模型"""
    print("🚀 快速訓練模式 - 僅訓練 45cm hybrid 模型")
    print("="*50)
    
    # 設定路徑
    mmwave_base_path = r"C:\Users\jk121\Documents\Code\mmWave-PAPER\Output\merged_csv"
    ecg_base_path = r"C:\Users\jk121\Documents\Code\mmWave-PAPER\Output\ECG Data\CSV"
    
    # 創建訓練管理器
    trainer = TrainingManager(mmwave_base_path, ecg_base_path)
    
    # 快速訓練設定
    results = trainer.train_all_models(
        distances=['45cm'],
        model_types=['hybrid'],
        max_pairs_per_distance=15,
        epochs=30,
        save_dir='mmWave_heart_amplitude/code/trained_models'
    )
    
    if results:
        print("✅ 快速訓練完成!")
        for model_key, result in results.items():
            print(f"  模型保存路徑: {result['save_path']}")
    else:
        print("❌ 快速訓練失敗")

# %%
if __name__ == "__main__":
    # 選擇運行模式
    print("選擇運行模式:")
    print("1. 完整訓練 (所有距離和模型類型)")
    print("2. 快速訓練 (僅45cm hybrid模型)")
    
    choice = input("請輸入選項 (1 或 2，預設為 2): ").strip() or "2"
    
    if choice == "1":
        main()
    else:
        quick_train()
