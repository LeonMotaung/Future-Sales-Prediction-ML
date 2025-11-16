# 📊 Future Sales Prediction – Machine Learning Web Application

This project is a **full-stack machine learning application** built using **Flask**, designed to predict **sales volume** based on multiple business factors such as:

- Advertising Spend  
- Promotions  
- Holidays  

Users can **train models on demand** directly from the frontend, choose between multiple ML algorithms, generate **ensemble models**, visualize the **confusion matrix**, and get real-time predictions.

---

## 🚀 Features

### ✅ 1. Train ML Models with One Click
Users can select:

- Random Forest  
- SVM  
- Naive Bayes  
- Logistic Regression  
- OR multiple algorithms → automatically trains **Ensemble Model**  
- Optionally include the **TCG Model** for more advanced predictions

### ✅ 2. Live Training Feedback
After training, the API returns:

- 📈 Accuracy  
- ⏱️ Training time  
- 🧾 Classification report  
- 🖼️ Confusion matrix image (`chart.png`)  
- ✔️ Success message

The `chart.png` visualizes the confusion matrix for the trained model.  

The `model.png` shows all four model performances together (Random Forest, Naive Bayes, SVM, Logistic Regression) and highlights the confusion matrix comparison.

### ✅ 3. Real-Time Prediction
Users can input:

- Advertising Spend  
- Number of Promotions  
- Number of Holidays  

The system returns the predicted sales category:

- **Low**  
- **Medium**  
- **High**

---

### ✅ 4. Persistent Model Storage
After training, the selected model is saved as:

