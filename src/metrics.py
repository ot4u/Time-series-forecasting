import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_and_forecast(column_name, series, steps=36):
    model_filename = os.path.join(models_dir, f'{column_name}_model.pkl')
    model = joblib.load(model_filename)
    forecast = model.get_forecast(steps=steps)
    return forecast.predicted_mean


input_file = '../datasets/view.csv'  
models_dir = '../models'

data = pd.read_csv(input_file, index_col='date', parse_dates=True)
data.replace('--', np.nan, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data.interpolate(method='time', inplace=True)

test_data = data[-36:]
train_data = data[:-36]

forecasts = {}

for column in train_data.columns:
    print(f"Прогнозирование для {column}...")
    forecast = load_and_forecast(column, train_data[column])
    forecasts[column] = forecast

forecast_df = pd.DataFrame(forecasts)
forecast_df.index = test_data.index

mae_scores = {}
mse_scores = {}

for column in test_data.columns:
    actual = test_data[column].dropna().values
    pred = forecast_df[column].values
    
    mae_scores[column] = mean_absolute_error(actual, pred)
    mse_scores[column] = mean_squared_error(actual, pred)

    print(f"\n{column}:")
    print(f"MAE: {mae_scores[column]}")
    print(f"MSE: {mse_scores[column]}")

average_mae = np.mean(list(mae_scores.values()))
average_mse = np.mean(list(mse_scores.values()))

print(f"\nСредние метрики по всем временным рядам:")
print(f"MAE: {average_mae}")
print(f"MSE: {average_mse}")
