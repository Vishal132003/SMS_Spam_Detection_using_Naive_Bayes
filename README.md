# 📩 SMS Spam Detection using Naive Bayes

## 📘 Overview
This project classifies SMS messages as **Spam** or **Ham (Not Spam)** using the **Naive Bayes** classification algorithm.  
The model learns from labeled SMS messages and predicts whether a new message is spam or not.

---

## 📊 Dataset
**Dataset:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

### 📁 Features
| Column | Description |
|---------|-------------|
| label | The category of the message — `ham` (not spam) or `spam` |
| message | The actual text content of the SMS |

---

## ⚙️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 🧠 Algorithm Used
**Naive Bayes (MultinomialNB)**  
Naive Bayes is a probabilistic classifier based on Bayes’ Theorem.  
It assumes independence between predictors and is especially effective for **text classification problems** like spam detection.

---

## 🔍 Steps Performed
1. **Importing Libraries**  
2. **Loading the Dataset**  
3. **Data Cleaning & Preprocessing**  
   - Dropped duplicate messages using `df.drop_duplicates(inplace=True)`  
   - Checked for null values  
   - Encoded labels (`ham` → 0, `spam` → 1)  
4. **Feature Extraction**  
   - Used **CountVectorizer** to convert text into numerical vectors  
5. **Splitting the Data**  
   - 80% training, 20% testing  
   - Used `random_state=42` for best accuracy and reproducibility  
6. **Model Training**  
   - Trained **MultinomialNB** model on training data  
7. **Model Evaluation**  
   - Accuracy  
   - Confusion Matrix  
   - Classification Report  
8. **Prediction on New Data**

---

## 💡 Example Prediction
```python
new_sms = ["Congratulations! You've won a $1000 gift card! Click to claim now."]
new_sms_vector = cv.transform(new_sms).toarray()
prediction = model.predict(new_sms_vector)

if prediction[0] == 1:
    print("🚨 Spam Message Detected!")
else:
    print("✅ This is a Ham (Not Spam) message.")
