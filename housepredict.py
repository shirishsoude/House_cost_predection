import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
data = pd.DataFrame({
    'area': [2600, 3000, 3200, 3600, 4000, 4100],
    'bedrooms': [3, 4, None, 3, 5, 6],
    'age': [20, 15, 18, 30, 8, 8],
    'price': [550000, 565000, 610000, 595000, 760000, 810000]
})

# Additional data
new_data = pd.DataFrame({
    'area': [2800, 3200, 2800, 3500, 3800, 4200],
    'bedrooms': [4, 3, 2, 4, 3, 5],
    'age': [22, 10, 12, 5, 6, 3],
    'price': [620000, 590000, 540000, 670000, 710000, 800000]
})

# Concatenate existing and new data
data = pd.concat([data, new_data], ignore_index=True)

# Step 2: Data Preprocessing
data = data.dropna()  # Drop rows with missing values

X = data.drop('price', axis=1)  # Features
y = data['price']  # Target variable

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Selection
model = LinearRegression()

# Step 5: Model Training
model.fit(X_train, y_train)

# Step 6: Model Evaluation
train_predictions = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)

test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)

# Step 7: Model Deployment
# Create Streamlit web application
def predict_price(area, bedrooms, age):
    new_data = pd.DataFrame({'area': [area], 'bedrooms': [bedrooms], 'age': [age]})
    predicted_price = model.predict(new_data)[0]
    return predicted_price

st.title("House Price Prediction")

# Input fields
area = st.number_input("Area (sq. ft.):", min_value=0)
bedrooms = st.number_input("Bedrooms:", min_value=0, max_value=10, step=1)
age = st.number_input("Age (years):", min_value=0)

# Prediction button
if st.button("Predict"):
    predicted_price = predict_price(area, bedrooms, age)
    predicted_price_usd = predicted_price * (data['price'].max() - data['price'].min()) + data['price'].min()
    st.success(f"The predicted house price is ${predicted_price_usd:.2f} USD")





