# 🧠 Multimodal Stress Detection Using Audio and Text

## 📌 Overview

This project presents a **multimodal machine learning system** for detecting stress by analyzing both **textual data (NLP)** and **audio signals (speech features)**. By combining multiple data modalities, the system achieves significantly higher accuracy and robustness compared to traditional single-modality approaches.

---

## 🚀 Key Highlights

* 🔤 Text-based stress detection using NLP techniques
* 🔊 Audio-based stress detection using speech signal processing
* 🔗 Multimodal fusion model combining both modalities
* 📊 Achieves up to **97% accuracy**
* 📈 Evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC

---

## 🧪 Methodology

### 🔹 Text-Based Model

* Preprocessing:

  * Lowercasing, stopword removal, punctuation cleaning
* Feature Extraction:

  * **TF-IDF (unigrams + bigrams)**
* Models:

  * Logistic Regression
  * Support Vector Machine (SVM)
  * Random Forest

---

### 🔹 Audio-Based Model

* Feature Extraction using **Librosa**:

  * MFCC (Mel-Frequency Cepstral Coefficients)
  * Chroma Features
  * Spectral Contrast
  * Tonnetz
* Model:

  * Support Vector Machine (SVM with RBF kernel)

---

### 🔹 Multimodal Fusion

* Combines predictions from text and audio models
* Uses **weighted late fusion** approach
* Enhances prediction accuracy and generalization

---

## 📊 Results

| Model       | Accuracy | AUC  |
| ----------- | -------- | ---- |
| Text Model  | ~84.6%   | 0.91 |
| Audio Model | ~95.5%   | 1.00 |
| Multimodal  | **~97%** | 0.99 |

👉 Multimodal approach significantly outperforms individual models.

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:**

  * scikit-learn
  * librosa
  * numpy, pandas
  * matplotlib, seaborn

---

## 📂 Project Structure

```
├── stacking.py              # Main pipeline
├── app.py                   # Application script (if applicable)
├── results/                 # Output visualizations
├── README.md
└── .gitignore
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/rupshamishra/MULTI-MODEL-STRESS-DETECTION-USING-TEXT-AND-AUDIO.git
```

2. Navigate to the project folder:

```
cd MULTI-MODEL-STRESS-DETECTION-USING-TEXT-AND-AUDIO
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the project:

```
python stacking.py
```

---

## 📁 Dataset

Due to size constraints, datasets are not included in this repository.

* Text data: Reddit & Twitter stress datasets
* Audio data: Stress speech dataset

👉 Datasets can be sourced from platforms like Kaggle or provided upon request.

---

## 💡 Applications

* Mental health monitoring systems
* Workplace stress detection
* Social media sentiment analysis
* AI-based healthcare solutions

---

## 🔮 Future Scope

* Integration of deep learning models (BERT, wav2vec)
* Addition of facial expression analysis (tri-modal system)
* Deployment as a real-time web application (Streamlit)
* Improved model explainability using SHAP

## 🎓 Institution

Kalinga Institute of Industrial Technology (KIIT), Bhubaneswar

---

## 📜 License

This project is intended for academic and research purposes only.


