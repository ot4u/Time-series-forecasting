import pandas as pd
import numpy as np
import joblib
import os
import sys

def load_and_forecast(column_name, series, steps=36):
    model_filename = os.path.join(models_dir, f'{column_name}_model.pkl')
    model = joblib.load(model_filename)
    forecast = model.get_forecast(steps=steps)
    return forecast.predicted_mean

input_file = '../datasets/view.csv' #укажите путь до вашего датасета  
output_file = '../datasets/output.csv'
models_dir = '../models'

data = pd.read_csv(input_file, index_col='date', parse_dates=True)
data.replace('--', np.nan, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data.interpolate(method='time', inplace=True)


forecasts = {}
for column in data.columns:
    print(f"Прогнозирование для {column}...")
    forecast = load_and_forecast(column, data[column])
    forecasts[column] = forecast

forecast_df = pd.DataFrame(forecasts)
forecast_df.index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=36, freq='MS')
result_df = pd.concat([data, forecast_df])
result_df.to_csv(output_file, index_label='date')

print(f"Прогнозирование завершено. Результаты сохранены в {output_file}.")
