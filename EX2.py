import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import librosa
import librosa.display
import matplotlib.pyplot as plt


# --- Загрузка данных ---
def load_wav_files(folder_path, labeled=True):
    data, labels = [], []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            path = os.path.join(folder_path, file_name)
            sampling_rate, signal = wavfile.read(path)
            data.append(signal)
            if labeled:
                label = int(file_name.split('_')[1].split('.')[0])
                labels.append(label)
    return (data, labels) if labeled else data


# --- Нормализация длины сигналов ---
def normalize_length(signals, target_length):
    return [
        np.pad(signal, (0, max(0, target_length - len(signal))), 'constant')[:target_length]
        for signal in signals
    ]


# --- Извлечение признаков временной области ---
def extract_features_advanced(signals):
    features = []
    for signal in signals:
        mean = np.mean(signal)
        std = np.std(signal)
        max_val = np.max(signal)
        min_val = np.min(signal)
        skewness = np.mean((signal - mean) ** 3) / (std ** 3)
        kurtosis = np.mean((signal - mean) ** 4) / (std ** 4)
        rms = np.sqrt(np.mean(signal**2))  # RMS
        features.append([mean, std, max_val, min_val, skewness, kurtosis, rms])
    return np.array(features)


# --- Извлечение признаков частотной области ---
def extract_frequency_features(signals):
    features = []
    for signal in signals:
        fft_signal = np.fft.fft(signal)
        power_spectrum = np.abs(fft_signal) ** 2
        mean_power = np.mean(power_spectrum)
        std_power = np.std(power_spectrum)
        features.append([mean_power, std_power])
    return np.array(features)


# --- Извлечение MFCC ---
def extract_mfcc_features(signals, sample_rate, n_mfcc=13):
    features = []
    for signal in signals:
        signal = signal.astype(np.float32)  # Преобразуем в float32
        signal = signal / np.max(np.abs(signal))  # Масштабируем в диапазон [-1, 1]
        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)
        features.append(np.mean(mfcc, axis=1))
    return np.array(features)


# --- Аугментация данных ---
def augment_data(signals):
    augmented_signals = []
    for signal in signals:
        noise = np.random.normal(0, 0.01, signal.shape)
        shifted = np.roll(signal, 1000)
        augmented_signals.extend([signal, signal + noise, shifted])
    return augmented_signals


# --- Загрузка и обработка данных ---
dev_folder = "./data/free-spoken-digit/dev"
eval_folder = "./data/free-spoken-digit/eval"

X_train_raw, y_train = load_wav_files(dev_folder)
X_eval_raw = load_wav_files(eval_folder, labeled=False)

target_length = max(len(signal) for signal in X_train_raw)
X_train_raw = normalize_length(X_train_raw, target_length)
X_eval_raw = normalize_length(X_eval_raw, target_length)

# Аугментация данных
X_train_raw = augment_data(X_train_raw)
y_train = y_train * 3  # Дублирование меток для новых данных

# Извлечение признаков
samplerate = 8000
X_train_features = extract_features_advanced(X_train_raw)
X_eval_features = extract_features_advanced(X_eval_raw)

freq_features_train = extract_frequency_features(X_train_raw)
freq_features_eval = extract_frequency_features(X_eval_raw)

mfcc_features_train = extract_mfcc_features(X_train_raw, samplerate)
mfcc_features_eval = extract_mfcc_features(X_eval_raw, samplerate)

X_train_features = np.hstack([X_train_features, freq_features_train, mfcc_features_train])
X_eval_features = np.hstack([X_eval_features, freq_features_eval, mfcc_features_eval])

# Нормализация
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_eval_features = scaler.transform(X_eval_features)

# Разделение на обучение и валидацию
X_train, X_val, y_train_split, y_val = train_test_split(X_train_features, y_train, test_size=0.2, random_state=42)

# --- Модель и подбор гиперпараметров ---
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20],
    'learning_rate': [0.01, 0.1, 0.2],
}
grid_search = GridSearchCV(estimator=XGBClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train_split)

best_clf = grid_search.best_estimator_

# Оценка модели
y_pred = best_clf.predict(X_val)
f1 = f1_score(y_val, y_pred, average='weighted')
accuracy = accuracy_score(y_val, y_pred)
print(f"Best F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

# --- Визуализация матрицы ошибок ---
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# --- Генерация предсказаний для теста ---
y_eval_pred = best_clf.predict(X_eval_features)

# --- Сохранение результатов ---
# eval_ids = [f.split('.')[0] for f in os.listdir(eval_folder) if f.endswith('.wav')]
# submission = pd.DataFrame({'Id': eval_ids, 'Predicted': y_eval_pred})

# submission['Id'] = submission['Id'].astype(int)
# submission = submission.sort_values(by='Id')
# submission.to_csv('submission.csv', index=False)

# print("Упорядоченный файл 'submission.csv' создан.")
