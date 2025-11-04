# 📩 SMS Spam Detection using Naive Bayes

## Overview
This project predicts whether an SMS message is spam or not using the Naive Bayes algorithm.  
The model is trained on a labeled dataset of text messages, where each message is categorized as either "ham" (not spam) or "spam".  
It helps automatically filter unwanted or promotional messages from genuine ones.

---

## Dataset
The dataset used for this project is the **SMS Spam Collection Dataset** from Kaggle (kaggle.com/datasets/uciml/sms-spam-collection-dataset).  
It contains two columns:
- `label`: The message type, either ham or spam.
- `message`: The actual content of the SMS.

---

## Technologies Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.

---

## Algorithm Used
Naive Bayes (MultinomialNB) was used for this project.  
It works on the principle of Bayes' Theorem and is particularly efficient for text-based classification problems like spam detection.  
Since the dataset contains word count data, MultinomialNB performs better than GaussianNB.

---

## Steps Performed
1. Imported the required libraries.
2. Loaded the dataset and checked its structure.
3. Removed duplicate rows using `df.drop_duplicates(inplace=True)` and verified there were no missing values.
4. Encoded the labels: 'ham' as 0 and 'spam' as 1.
5. Converted text into numeric form using CountVectorizer.
6. Split the dataset into training and testing sets with a test size of 20% and random_state=42.
7. Trained the MultinomialNB model on the training data.
8. Evaluated the model using accuracy score, confusion matrix, and classification report.
9. Tested the model on new input messages.

---

## Example Prediction
```python
new_sms = ["Congratulations! You've won a $1000 gift card! Click to claim now."]
new_sms_vector = cv.transform(new_sms).toarray()
prediction = model.predict(new_sms_vector)

if prediction[0] == 1:
    print("🚨 Spam Message Detected!")
else:
    print("✅ This is a Ham (Not Spam) message.")
