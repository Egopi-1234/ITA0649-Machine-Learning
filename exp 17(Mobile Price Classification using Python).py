import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = {
    'battery_power': [1200, 2000, 1500, 3000, 4000],
    'ram': [512, 1024, 2048, 3072, 4096],  # in MB
    'internal_memory': [8, 16, 32, 64, 128],  # in GB
    'screen_size': [5.0, 5.5, 6.0, 6.5, 7.0],  # in inches
    'camera_quality': [8, 12, 16, 20, 32],  # in MP
    'price_range': [0, 1, 2, 3, 3]  # Categorical values: 0 (low), 1 (mid-low), 2 (mid-high), 3 (high)
}
df = pd.DataFrame(data)
print("Sample DataFrame:")
print(df)
X = df.drop('price_range', axis=1)  
y = df['price_range'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
loo = LeaveOneOut() 
cross_val_scores = cross_val_score(model, X, y, cv=loo)  
print(f"\nLeave-One-Out Cross-Validation Scores: {cross_val_scores}")
print(f"Average Leave-One-Out Cross-Validation Score: {cross_val_scores.mean()}")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
new_data = pd.DataFrame({
    'battery_power': [3500],
    'ram': [2048],
    'internal_memory': [64],
    'screen_size': [6.0],
    'camera_quality': [16]
})
new_data_scaled = scaler.transform(new_data)
new_prediction = model.predict(new_data_scaled)
print("\nPredicted Price Range for new data:", new_prediction[0])
