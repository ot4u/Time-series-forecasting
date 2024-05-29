import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import os

def train_sarima_model(series, column_name):
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)
    model_filename = os.path.join('../models', f'{column_name}_model.pkl')
    joblib.dump(results, model_filename)


data = pd.read_csv('../datasets/view.csv', index_col='date', parse_dates=True)
data.replace('--', np.nan, inplace=True)
data = data.apply(pd.to_numeric, errors = 'coerce')
data.interpolate(method='time', inplace=True)

for column in data.columns:
    print(f"Обучение модели для {column}...")
    train_sarima_model(data[column], column)

print("Обучение завершено и модели сохранены.")