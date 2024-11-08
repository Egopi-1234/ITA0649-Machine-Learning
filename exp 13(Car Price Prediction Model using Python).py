import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
data = {
    'brand': ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes'],
    'model': ['Corolla', 'Civic', 'Focus', '3 Series', 'C Class'],
    'year': [2015, 2016, 2018, 2019, 2020],
    'mileage': [50000, 40000, 30000, 20000, 15000],
    'fuel_type': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol'],
    'engine_size': [1.8, 1.5, 2.0, 2.5, 3.0],
    'price': [15000, 16000, 17000, 30000, 35000]
}
df = pd.DataFrame(data)
df = pd.get_dummies(df, drop_first=True)
X = df.drop('price', axis=1)  
y = df['price']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Evaluation Metrics:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
with open('car_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("\nModel saved as 'car_price_model.pkl'.")

with open('car_price_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
new_car_data = pd.DataFrame({
    'year': [2021],
    'mileage': [25000],
    'engine_size': [1.6],
    'fuel_type_Diesel': [0],  
    'fuel_type_Petrol': [1],
    'brand_Toyota': [0],  
    'brand_Honda': [0],
    'brand_Ford': [0],
    'brand_BMW': [0],
    'brand_Mercedes': [0],
    'model_3 Series': [0],  
    'model_C Class': [0],
    'model_Civic': [0],
    'model_Focus': [0],
    'model_Corolla': [1]  
})
new_car_data = new_car_data[X_train.columns]  
predicted_price = loaded_model.predict(new_car_data)
print("\nPredicted price for the new car:", predicted_price)
