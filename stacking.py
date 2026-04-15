# ==========================================
# MULTIMODAL STRESS DETECTION SYSTEM
# FINAL VERSION (STACKING + AUDIO + FUSION)
# ==========================================

import pandas as pd
import numpy as np
import string
import re
import nltk
import librosa
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectPercentile, chi2

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

print("\n========================================")
print("MULTIMODAL STRESS DETECTION SYSTEM")
print("========================================")

# ==========================================
# LOAD TEXT DATA
# ==========================================

df1 = pd.read_csv("Reddit1.csv.csv", sep=";", engine="python", on_bad_lines="skip")
df2 = pd.read_csv("Reddit2.csv.csv", sep=";", engine="python", on_bad_lines="skip")
df3 = pd.read_csv("Twitter1.csv.csv", sep=";", engine="python", on_bad_lines="skip")
df4 = pd.read_csv("Twitter2.csv.csv", sep=";", engine="python", on_bad_lines="skip")

text_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

for col in ["text", "body", "content", "post", "title"]:
    if col in text_df.columns:
        text_df.rename(columns={col: "text"}, inplace=True)
        break

if "label" in text_df.columns:
    text_df.rename(columns={"label": "Stress_Level"}, inplace=True)

text_df = text_df.dropna(subset=["text", "Stress_Level"])

le = LabelEncoder()

# ==========================================
# TEXT CLEANING
# ==========================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and 3 <= len(w) <= 18
    ]
    return " ".join(words)

print("Cleaning text...")
text_df["cleaned_text"] = text_df["text"].apply(clean_text)
text_df = text_df[text_df["cleaned_text"].str.strip() != ""]

y_text = le.fit_transform(text_df["Stress_Level"])

# ==========================================
# TF-IDF + FEATURE SELECTION
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1,2),
    min_df=5,
    max_df=0.85
)

X = vectorizer.fit_transform(text_df["cleaned_text"])

selector = SelectPercentile(chi2, percentile=40)
X = selector.fit_transform(X, y_text)

# ==========================================
# SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_text,
    test_size=0.2,
    random_state=42,
    stratify=y_text
)

# ==========================================
# STACKING TEXT MODEL
# ==========================================

print("\nTraining Text Model (Stacking)...")

base_models = [
    ("lr", LogisticRegression(max_iter=1000, C=0.5)),
    ("rf", RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )),
    ("svm", SVC(probability=True, C=0.8))
]

text_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(C=0.3),
    cv=5
)

text_model.fit(X_train, y_train)

# TEXT METRICS
text_train_acc = accuracy_score(y_train, text_model.predict(X_train))
text_preds = text_model.predict(X_test)
text_probs = text_model.predict_proba(X_test)[:,1]
text_test_acc = accuracy_score(y_test, text_preds)

print("\nTEXT TRAIN ACCURACY:", text_train_acc)
print("TEXT TEST ACCURACY:", text_test_acc)

# ==========================================
# TEXT CONFUSION MATRIX
# ==========================================

plt.figure()
sns.heatmap(confusion_matrix(y_test, text_preds), annot=True, fmt="d")
plt.title("Text Confusion Matrix")
plt.show()

fpr_text, tpr_text, _ = roc_curve(y_test, text_probs)
roc_text = auc(fpr_text, tpr_text)

# ==========================================
# AUDIO MODEL
# ==========================================

audio_df = pd.read_csv("audio_dataset.csv")

def extract_features(file):
    try:
        y, sr = librosa.load(file, duration=5)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spectral = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

        return np.hstack([mfcc, chroma, spectral, zcr, rms, tonnetz])

    except:
        return np.zeros(201)

print("\nExtracting audio features...")

X_audio = np.array([extract_features(f) for f in audio_df["file_path"]])
y_audio = le.fit_transform(audio_df["Stress_Level"])

Xa_train, Xa_test, ya_train, ya_test = train_test_split(
    X_audio, y_audio,
    test_size=0.2,
    random_state=42,
    stratify=y_audio
)

scaler = StandardScaler()
Xa_train = scaler.fit_transform(Xa_train)
Xa_test = scaler.transform(Xa_test)

audio_model = SVC(kernel="rbf", probability=True, C=0.8)
audio_model.fit(Xa_train, ya_train)

audio_train_acc = accuracy_score(ya_train, audio_model.predict(Xa_train))
audio_preds = audio_model.predict(Xa_test)
audio_probs = audio_model.predict_proba(Xa_test)[:,1]
audio_test_acc = accuracy_score(ya_test, audio_preds)

print("\nAUDIO TRAIN ACCURACY:", audio_train_acc)
print("AUDIO TEST ACCURACY:", audio_test_acc)

# ==========================================
# AUDIO CONFUSION MATRIX
# ==========================================

plt.figure()
sns.heatmap(confusion_matrix(ya_test, audio_preds), annot=True, fmt="d")
plt.title("Audio Confusion Matrix")
plt.show()

fpr_audio, tpr_audio, _ = roc_curve(ya_test, audio_probs)
roc_audio = auc(fpr_audio, tpr_audio)

# ==========================================
# MULTIMODAL FUSION
# ==========================================

min_len = min(len(text_probs), len(audio_probs))

text_probs = text_probs[:min_len]
audio_probs = audio_probs[:min_len]
y_true = ya_test[:min_len]

fusion_probs = 0.3 * text_probs + 0.7 * audio_probs
fusion_preds = (fusion_probs >= 0.5).astype(int)

multi_acc = accuracy_score(y_true, fusion_preds)

print("\nMULTIMODAL ACCURACY:", multi_acc)

# ==========================================
# MULTIMODAL CONFUSION MATRIX
# ==========================================

plt.figure()
sns.heatmap(confusion_matrix(y_true, fusion_preds), annot=True, fmt="d")
plt.title("Multimodal Confusion Matrix")
plt.show()

fpr_multi, tpr_multi, _ = roc_curve(y_true, fusion_probs)
roc_multi = auc(fpr_multi, tpr_multi)

# ==========================================
# ROC COMPARISON
# ==========================================

plt.figure()

plt.plot(fpr_text, tpr_text, label="Text (AUC = %0.2f)" % roc_text)
plt.plot(fpr_audio, tpr_audio, label="Audio (AUC = %0.2f)" % roc_audio)
plt.plot(fpr_multi, tpr_multi, label="Multimodal (AUC = %0.2f)" % roc_multi)

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Comparison")

plt.legend()
plt.show()

# ==========================================
# SAVE MODELS
# ==========================================

joblib.dump(text_model, "text_model.pkl")
joblib.dump(audio_model, "audio_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(selector, "selector.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ FINAL MODEL READY")