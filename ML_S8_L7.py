from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.exceptions import ConvergenceWarning  # Import ConvergenceWarning
import pandas as pd
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Check if optional libraries are available
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# Function to load and preprocess data

def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)
    
    # Drop categorical columns
    data = data.select_dtypes(exclude=['object', 'category'])
   
    X = data.drop('confidence', axis=1)
    y = data['confidence']
    return X, y


# Function to perform cross-validation and hyperparameter tuning using RandomizedSearchCV
def tune_hyperparameters(X, y, model, param_distributions, n_iter=10):  # Reduce n_iter to speed up
    search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, scoring='accuracy', cv=5, random_state=42)
    search.fit(X, y)
    return search.best_estimator_, search.best_score_

# Function to evaluate and compare various classifiers
def evaluate_classifiers(X, y):
    classifiers = {
        'Perceptron': Perceptron(),
        'MLP': MLPClassifier(max_iter=1000),  # Increase max_iter to avoid convergence issues
        'SVM': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Naive Bayes': GaussianNB()
    }

    # Add XGBoost if available
    if XGBClassifier is not None:
        classifiers['XGBoost'] = XGBClassifier()

    # Add CatBoost if available
    if CatBoostClassifier is not None:
        classifiers['CatBoost'] = CatBoostClassifier()

    results = []
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')  # Reduce cv to 3 for faster evaluation
        results.append({
            'Model': name,
            'Accuracy': np.mean(scores),
            'StdDev': np.std(scores)
        })
    return pd.DataFrame(results)

# Function to calculate detailed performance metrics
def calculate_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted')
    }

# Main program
def main():
    file_path = r"C:\Users\shrey\OneDrive\Desktop\Sem 5\Machine Learning\Feature Extraction using TF-IDF.xlsx"
    X, y = load_and_preprocess_data(file_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning for Perceptron model
    perceptron_params = {'penalty': ['l2', 'elasticnet'], 'alpha': [1e-4, 1e-3, 1e-2, 1e-1]}
    best_perceptron, best_score_perceptron = tune_hyperparameters(X_train, y_train, Perceptron(), perceptron_params, n_iter=8)
    
    # Hyperparameter tuning for MLP (Multi-Layer Perceptron) model
    mlp_params = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['tanh', 'relu'], 'alpha': [1e-4, 1e-3]}
    best_mlp, best_score_mlp = tune_hyperparameters(X_train, y_train, MLPClassifier(max_iter=1000), mlp_params, n_iter=8)
    
    # Evaluate various classifiers
    evaluation_results = evaluate_classifiers(X_train, y_train)
    
    # Train and evaluate the best Perceptron model on the test set
    best_perceptron.fit(X_train, y_train)
    perceptron_predictions = best_perceptron.predict(X_test)
    perceptron_metrics = calculate_metrics(y_test, perceptron_predictions)
    
    # Train and evaluate the best MLP model on the test set
    best_mlp.fit(X_train, y_train)
    mlp_predictions = best_mlp.predict(X_test)
    mlp_metrics = calculate_metrics(y_test, mlp_predictions)
    
    # Display the results of hyperparameter tuning and model evaluation
    print("Hyperparameter Tuning Results:")
    print("Best Perceptron Model:", best_perceptron)
    print("Best Perceptron Score:", best_score_perceptron)
    print("Best MLP Model:", best_mlp)
    print("Best MLP Score:", best_score_mlp)
    
    print("\nDetailed Performance Metrics for Best Perceptron Model on Test Set:")
    print(perceptron_metrics)
    
    print("\nDetailed Performance Metrics for Best MLP Model on Test Set:")
    print(mlp_metrics)
    
    print("\nClassifier Evaluation Results:")
    print(evaluation_results)

# Run the main program
if __name__ == "__main__":
    main()