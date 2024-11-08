import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
np.random.seed(42)
n_samples = 1000

data = {
    'Income': np.random.randint(20000, 100000, n_samples),
    'Debt': np.random.randint(1000, 50000, n_samples),
    'Age': np.random.randint(18, 70, n_samples),
    'Payment History': np.random.choice(['Good', 'Average', 'Poor'], n_samples),
    'Credit Score': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor'], n_samples)
}

df = pd.DataFrame(data)
label_encoder = LabelEncoder()
df['Payment History'] = label_encoder.fit_transform(df['Payment History'])
df['Credit Score'] = label_encoder.fit_transform(df['Credit Score'])  # Target variable
X = df[['Income', 'Debt', 'Age', 'Payment History']]
y = df['Credit Score']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.best_estimator_.predict(X_test)
print("Best Parameters:", clf.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
new_data = np.array([[45000, 15000, 30, label_encoder.transform(['Good'])[0]]])
new_data_df = pd.DataFrame(new_data, columns=X.columns)  # Fixing feature name warning
prediction = clf.predict(new_data_df)
print("\nPredicted Credit Score Class:", label_encoder.inverse_transform(prediction)[0])
