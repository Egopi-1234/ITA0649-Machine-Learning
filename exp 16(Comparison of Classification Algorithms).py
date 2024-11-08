import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import pandas as pd
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss')  
}
results = []
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "F1-Score": f1,
        "Training Time (s)": training_time
    })
results_df = pd.DataFrame(results)
print("Model Performance Comparison:\n")
print(results_df)
plt.figure(figsize=(10, 6))
sns.barplot(x="Accuracy", y="Model", data=results_df, hue="Model", dodge=False)
plt.title("Model Accuracy Comparison")
plt.xlabel("Accuracy")
plt.ylabel("Model")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
plt.figure(figsize=(10, 6))
sns.barplot(x="F1-Score", y="Model", data=results_df, hue="Model", dodge=False)
plt.title("Model F1-Score Comparison")
plt.xlabel("F1-Score")
plt.ylabel("Model")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
plt.figure(figsize=(10, 6))
sns.barplot(x="Training Time (s)", y="Model", data=results_df, hue="Model", dodge=False)
plt.title("Model Training Time Comparison")
plt.xlabel("Training Time (seconds)")
plt.ylabel("Model")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
