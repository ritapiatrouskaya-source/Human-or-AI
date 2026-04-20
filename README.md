# Human vs AI Text Detection

This project focuses on classifying whether a text is human-written or AI-generated using classical machine learning techniques.

---

## 📊 Dataset

* Training dataset:
  https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

* External evaluation dataset (for generalization):
  https://www.kaggle.com/datasets/shamimhasan8/ai-vs-human-text-dataset

Binary classification:

* **1** — AI-generated
* **0** — Human-written

---

## ⚙️ Approach

* Text preprocessing using **Bag-of-Words (CountVectorizer)**
* Model training with **XGBoost**
* Evaluation using **Accuracy, Precision, Recall, F1-score**
* Generalization tested on an external dataset
* Model calibration via **threshold tuning**

---

## 🔬 Experiment Tracking (MLflow)

MLflow was used to manage and compare multiple experiments:

* Tracked different model configurations and hyperparameters
* Compared performance across runs
* Selected the best model based on F1-score and generalization
* Stored artifacts for reproducibility

This enabled a structured and reproducible experimentation process.

---

## 📈 Results

**Initial model:**

* Recall = **1.0**
* Precision ≈ **0.5**
* Model overpredicted AI-generated texts (overconfident behavior)

**After threshold tuning:**

* Accuracy ≈ **0.88**
* F1-score ≈ **0.89**
* Improved balance between precision and recall

👉 The model was calibrated by adjusting the classification threshold (~0.985), significantly reducing false positives while maintaining strong detection capability.

---

## 📊 Model Calibration

The relationship between precision and recall across thresholds:

* <img width="1013" height="627" alt="image" src="https://github.com/user-attachments/assets/4d1a811d-edfc-41d7-ab20-2bf020226f1e" />


This analysis highlights how threshold tuning improves model reliability and real-world applicability.

---

## 🚀 Features

* End-to-end ML pipeline
* Feature importance analysis
* Evaluation on unseen data
* Model calibration via threshold tuning

---

## 🧪 Tech Stack

* Python
* Scikit-learn
* XGBoost
* MLflow
* Streamlit

---

## 🌐 Demo

👉 https://human-or-ai-irgw9pjesozfdmwaqn8rfp.streamlit.app/

Interactive app for real-time text classification.

---

## 📦 Notes

* Dataset is not included due to size limitations
* MLflow artifacts are excluded






