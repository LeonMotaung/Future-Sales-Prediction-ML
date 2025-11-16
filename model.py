import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from tcg_model import TCGClassifierWrapper

def get_model_for_algorithm(algorithm):
    if algorithm == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == 'svm':
        # Enable probability for soft voting in the ensemble
        return SVC(kernel='linear', random_state=42, probability=True)
    elif algorithm == 'naive_bayes':
        return GaussianNB()
    elif algorithm == 'logistic_regression':
        return LogisticRegression(random_state=42)
    elif algorithm == 'tcg':
        return TCGClassifierWrapper()
    else:
        raise ValueError(f"Unknown algorithm specified: {algorithm}")

def _generate_data():
    np.random.seed(42)
    n_samples = 1000
    advertising_spend = np.random.rand(n_samples) * 100
    promotions = np.random.randint(0, 2, n_samples)
    holidays = np.random.randint(0, 2, n_samples)
    
    sales_volume_score = (advertising_spend * 0.6 + promotions * 20 + holidays * 15 + np.random.randn(n_samples) * 10)
    
    bins = np.percentile(sales_volume_score, [33, 66])
    sales_volume = np.digitize(sales_volume_score, bins=bins)
    
    X = pd.DataFrame({
        'advertising_spend': advertising_spend,
        'promotions': promotions,
        'holidays': holidays
    })
    y = sales_volume
    return X, y

def _train_and_evaluate(model, X_train, X_test, y_train, y_test):
    target_names = ['Low', 'Medium', 'High']
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    training_time = round(end_time - start_time, 2)
    
    y_pred = model.predict(X_test)
    
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if not os.path.exists('static'):
        os.makedirs('static')
    
    plot_path = 'static/confusion_matrix.png'
    plt.savefig(plot_path)
    plt.close()
    
    return model, accuracy, training_time, class_report, plot_path

def train_selected_model(algorithm):
    X, y = _generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = get_model_for_algorithm(algorithm)
    return _train_and_evaluate(model, X_train, X_test, y_train, y_test)

def train_ensemble_model(algorithms):
    X, y = _generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    estimators = []
    for alg in algorithms:
        # Use a short name for the estimator tuple
        short_name = ''.join(word[0] for word in alg.split('_'))
        estimators.append((short_name, get_model_for_algorithm(alg)))
    
    ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
    
    return _train_and_evaluate(ensemble_model, X_train, X_test, y_train, y_test)
