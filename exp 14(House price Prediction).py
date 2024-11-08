import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
data = {
    'Location': ['New York', 'Los Angeles', 'San Francisco', 'Austin', 'Chicago'],
    'Size (sq ft)': [1200, 1500, 1800, 1400, 1600],
    'Bedrooms': [2, 3, 3, 2, 3],
    'Bathrooms': [1, 2, 2, 2, 2],
    'Age of House (years)': [10, 15, 5, 20, 8],
    'Price ($)': [500000, 650000, 700000, 400000, 450000]
}
df = pd.DataFrame(data)
label_encoder = LabelEncoder()
df['Location'] = label_encoder.fit_transform(df['Location'])
X = df.drop('Price ($)', axis=1) 
y = df['Price ($)']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)

print("\nDataset:")
print(df)
new_house = pd.DataFrame({
    'Location': label_encoder.transform(['Los Angeles']),  
    'Size (sq ft)': [1500],
    'Bedrooms': [3],
    'Bathrooms': [2],
    'Age of House (years)': [10]
})

predicted_price = model.predict(new_house)
print("\nPredicted price for the new house: $", predicted_price[0])
