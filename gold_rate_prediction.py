
# GOLD PRICE PREDICTION - ML - JEEVAN RAM K

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
def create_sample_data(n_samples=1000):
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    data = {
        'SPX': np.cumsum(np.random.randn(n_samples) * 2) + 3000,  
        'USO': np.abs(np.cumsum(np.random.randn(n_samples) * 1) + 60),  
        'SLV': np.cumsum(np.random.randn(n_samples) * 0.3) + 20,   
        'EUR/USD': np.cumsum(np.random.randn(n_samples) * 0.005) + 1.10  
    }
    df = pd.DataFrame(data)

    df['Gold_Price'] = (
        1800 - 0.2 * df['SPX'] + 2.5 * df['USO'] +
        15 * df['SLV'] - 500 * df['EUR/USD'] + np.random.randn(n_samples) * 10
    )
    return df

df = create_sample_data()
print("Dataset shape:", df.shape)
print(df.head())

X = df[['SPX', 'USO', 'SLV', 'EUR/USD']]
y = df['Gold_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\n Model trained successfully!")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.2f}")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "gold_model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("\n Model and Scaler saved successfully!")
print(f"Model path: {model_path}")
print(f"Scaler path: {scaler_path}")

sample_input = np.array([[4000, 70, 25, 1.10]])  
sample_scaled = scaler.transform(sample_input)
predicted_price = model.predict(sample_scaled)[0]
print(f"\nSample Prediction for Gold Price: ${predicted_price:.2f}")