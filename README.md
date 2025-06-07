# 🏠 House Price Prediction

This project predicts house prices using simple regression techniques.  
We use features like average number of rooms, lower income population percentage, and student-teacher ratio to predict the median value of homes in Boston.

---

## ✅ Project Overview

- **Goal:** Predict house prices based on key housing features.
- **Dataset:** [Boston Housing Dataset](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv)
- **Tech Stack:** Python, Pandas, scikit-learn

---

## 🔍 Features Used

- `rm` – Average number of rooms per dwelling  
- `lstat` – Percentage of lower status population  
- `ptratio` – Pupil-teacher ratio by town  

---

## 📊 Model

- **Algorithm:** Linear Regression
- **Evaluation Metric:** R² Score

---

## 🛠️ How to Run

1. Clone the repository
2. Make sure you have Python installed and run:

```bash
pip install pandas scikit-learn
python house_price_prediction.py
