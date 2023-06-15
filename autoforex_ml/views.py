from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
import json
import requests
import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta
import tensorflow as tf
import numpy as np
import pandas as pd


def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')

series = pd.read_csv('./currency_data_10_years.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)
currency = series.columns.tolist()

time = series.index

def make_future_forecast(values, model, into_future, window_size) -> list:
    """
    Make future forecasts into_future steps after value ends.

    Returns future forecasts as a list of floats.
    """
    # 2. Create an empty list for future forecasts/prepare data to forecast on
    future_forecast = []
    last_window = values[-window_size:]

    # 3. Make INTO_FUTURE number of predictions, altering the data which gets predicted on each
    for _ in range(into_future):
        # Predict on the last window then append it again, again, again (our model will eventually start to make forecasts on its own forecasts)
        future_pred = model.predict(tf.expand_dims(last_window, axis=0))
        # print(f"Predicting on:\n {last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n")

        # Append preds to future_forecast
        future_forecast.append(tf.squeeze(future_pred).numpy())

        # Update last window with new pred and get WINDOW_SIZE most recent preds (model was trained on WINDOW_SIZE windows)
        last_window = np.append(last_window, future_pred)[-window_size:]

    return future_forecast

def getPredictEUR(request):
    if request.method == 'GET':

        into_future = 6
        window_size = 30
        seriesEUR = series.iloc[:, 1]

        model_number = 22
        model = tf.keras.models.load_model(f"autoforex_ml/saved_model/['EUR']_model_1.h5")

        future_forecast = make_future_forecast(values=seriesEUR,
                                               model=model,
                                               into_future=into_future,
                                               window_size=window_size)

        next_time_steps = pd.date_range(start=time[-1], periods=into_future + 1)

        future_forecast = np.insert(future_forecast, 0, seriesEUR[-1])

        percentage = (future_forecast[-1]-future_forecast[0]) / future_forecast[0] * 100
        percentage = np.round(percentage, 2)
        # Create a dictionary with the desired format
        response_data = {'value': percentage}

        # Convert the dictionary to JSON
        json_response = json.dumps(response_data)

        # Set the content type to application/json
        return HttpResponse(json_response, content_type='application/json')
    

def getPredictUSD(request):
    if request.method == 'GET':

        into_future = 6
        window_size = 30
        seriesUSD = series.iloc[:, 2]

        model_number = 22
        model = tf.keras.models.load_model(f"autoforex_ml/saved_model/['USD']_model_1.h5")

        future_forecast = make_future_forecast(values=seriesUSD,
                                               model=model,
                                               into_future=into_future,
                                               window_size=window_size)

        next_time_steps = pd.date_range(start=time[-1], periods=into_future + 1)

        future_forecast = np.insert(future_forecast, 0, seriesUSD[-1])

        percentage = (future_forecast[-1]-future_forecast[0]) / future_forecast[0] * 100
        percentage = np.round(percentage, 2)
        # Create a dictionary with the desired format
        response_data = {'value': percentage}

        # Convert the dictionary to JSON
        json_response = json.dumps(response_data)

        # Set the content type to application/json
        return HttpResponse(json_response, content_type='application/json')
    
def getPredictSGD(request):
    if request.method == 'GET':

        into_future = 6
        window_size = 30
        seriesSGD = series.iloc[:, 5]

        model_number = 22
        model = tf.keras.models.load_model(f"autoforex_ml/saved_model/['SGD']_model_1.h5")

        future_forecast = make_future_forecast(values=seriesSGD,
                                               model=model,
                                               into_future=into_future,
                                               window_size=window_size)

        next_time_steps = pd.date_range(start=time[-1], periods=into_future + 1)

        future_forecast = np.insert(future_forecast, 0, seriesSGD[-1])

        percentage = (future_forecast[-1]-future_forecast[0]) / future_forecast[0] * 100
        percentage = np.round(percentage, 2)
        # Create a dictionary with the desired format
        response_data = {'value': percentage}

        # Convert the dictionary to JSON
        json_response = json.dumps(response_data)

        # Set the content type to application/json
        return HttpResponse(json_response, content_type='application/json')
    
def getPredictAUD(request):
    if request.method == 'GET':

        into_future = 6
        window_size = 30
        seriesAUD = series.iloc[:, 6]

        model_number = 22
        model = tf.keras.models.load_model(f"autoforex_ml/saved_model/['AUD']_model_1.h5")

        future_forecast = make_future_forecast(values=seriesAUD,
                                               model=model,
                                               into_future=into_future,
                                               window_size=window_size)

        next_time_steps = pd.date_range(start=time[-1], periods=into_future + 1)

        future_forecast = np.insert(future_forecast, 0, seriesAUD[-1])

        percentage = (future_forecast[-1]-future_forecast[0]) / future_forecast[0] * 100
        percentage = np.round(percentage, 2)
        # Create a dictionary with the desired format
        response_data = {'value': percentage}

        # Convert the dictionary to JSON
        json_response = json.dumps(response_data)

        # Set the content type to application/json
        return HttpResponse(json_response, content_type='application/json')


def getPredictCAD(request):
    if request.method == 'GET':

        into_future = 6
        window_size = 30
        seriesCAD = series.iloc[:, 8]

        model_number = 22
        model = tf.keras.models.load_model(f"autoforex_ml/saved_model/['CAD']_model_1.h5")

        future_forecast = make_future_forecast(values=seriesCAD,
                                               model=model,
                                               into_future=into_future,
                                               window_size=window_size)

        next_time_steps = pd.date_range(start=time[-1], periods=into_future + 1)

        future_forecast = np.insert(future_forecast, 0, seriesCAD[-1])

        percentage = (future_forecast[-1]-future_forecast[0]) / future_forecast[0] * 100
        percentage = np.round(percentage, 2)
        # Create a dictionary with the desired format
        response_data = {'value': percentage}

        # Convert the dictionary to JSON
        json_response = json.dumps(response_data)

        # Set the content type to application/json
        return HttpResponse(json_response, content_type='application/json')
    
def getPredictCNY(request):
    if request.method == 'GET':

        into_future = 6
        window_size = 30
        seriesCNY = series.iloc[:, 7]

        model_number = 22
        model = tf.keras.models.load_model(f"autoforex_ml/saved_model/['CNY']_model_1.h5")

        future_forecast = make_future_forecast(values=seriesCNY,
                                               model=model,
                                               into_future=into_future,
                                               window_size=window_size)

        next_time_steps = pd.date_range(start=time[-1], periods=into_future + 1)

        future_forecast = np.insert(future_forecast, 0, seriesCNY[-1])

        percentage = (future_forecast[-1]-future_forecast[0]) / future_forecast[0] * 100
        percentage = np.round(percentage, 2)
        # Create a dictionary with the desired format
        response_data = {'value': percentage}

        # Convert the dictionary to JSON
        json_response = json.dumps(response_data)

        # Set the content type to application/json
        return HttpResponse(json_response, content_type='application/json')
    
def getPredictGBP(request):
    if request.method == 'GET':

        into_future = 6
        window_size = 30
        seriesGBP = series.iloc[:, 4]

        model_number = 22
        model = tf.keras.models.load_model(f"autoforex_ml/saved_model/['GBP']_model_1.h5")

        future_forecast = make_future_forecast(values=seriesGBP,
                                               model=model,
                                               into_future=into_future,
                                               window_size=window_size)

        next_time_steps = pd.date_range(start=time[-1], periods=into_future + 1)

        future_forecast = np.insert(future_forecast, 0, seriesGBP[-1])

        percentage = (future_forecast[-1]-future_forecast[0]) / future_forecast[0] * 100
        percentage = np.round(percentage, 2)
        # Create a dictionary with the desired format
        response_data = {'value': percentage}

        # Convert the dictionary to JSON
        json_response = json.dumps(response_data)

        # Set the content type to application/json
        return HttpResponse(json_response, content_type='application/json')
    
def getPredictJPY(request):
    if request.method == 'GET':

        into_future = 6
        window_size = 30
        seriesJPY = series.iloc[:, 3]

        model_number = 22
        model = tf.keras.models.load_model(f"autoforex_ml/saved_model/['JPY']_model_1.h5")

        future_forecast = make_future_forecast(values=seriesJPY,
                                               model=model,
                                               into_future=into_future,
                                               window_size=window_size)

        next_time_steps = pd.date_range(start=time[-1], periods=into_future + 1)

        future_forecast = np.insert(future_forecast, 0, seriesJPY[-1])

        percentage = (future_forecast[-1]-future_forecast[0]) / future_forecast[0] * 100
        percentage = np.round(percentage, 2)
        # Create a dictionary with the desired format
        response_data = {'value': percentage}

        # Convert the dictionary to JSON
        json_response = json.dumps(response_data)

        # Set the content type to application/json
        return HttpResponse(json_response, content_type='application/json')
    
def getPredictMYR(request):
    if request.method == 'GET':

        into_future = 6
        window_size = 30
        seriesMYR = series.iloc[:, 9]

        model_number = 22
        model = tf.keras.models.load_model(f"autoforex_ml/saved_model/['MYR']_model_1.h5")

        future_forecast = make_future_forecast(values=seriesMYR,
                                               model=model,
                                               into_future=into_future,
                                               window_size=window_size)

        next_time_steps = pd.date_range(start=time[-1], periods=into_future + 1)

        future_forecast = np.insert(future_forecast, 0, seriesMYR[-1])

        percentage = (future_forecast[-1]-future_forecast[0]) / future_forecast[0] * 100
        percentage = np.round(percentage, 2)
        # Create a dictionary with the desired format
        response_data = {'value': percentage}

        # Convert the dictionary to JSON
        json_response = json.dumps(response_data)

        # Set the content type to application/json
        return HttpResponse(json_response, content_type='application/json')
    
def getPredictRUB(request):
    if request.method == 'GET':

        into_future = 6
        window_size = 30
        seriesRUB = series.iloc[:, 10]

        model_number = 22
        model = tf.keras.models.load_model(f"autoforex_ml/saved_model/['RUB']_model_1.h5")

        future_forecast = make_future_forecast(values=seriesRUB,
                                               model=model,
                                               into_future=into_future,
                                               window_size=window_size)

        next_time_steps = pd.date_range(start=time[-1], periods=into_future + 1)

        future_forecast = np.insert(future_forecast, 0, seriesRUB[-1])

        percentage = (future_forecast[-1]-future_forecast[0]) / future_forecast[0] * 100
        percentage = np.round(percentage, 2)
        # Create a dictionary with the desired format
        response_data = {'value': percentage}

        # Convert the dictionary to JSON
        json_response = json.dumps(response_data)

        # Set the content type to application/json
        return HttpResponse(json_response, content_type='application/json')

def callData(request):
    if request.method == 'GET':
        today = datetime.today()
        new_today_date = today.strftime("%Y-%m-%d")
        today = today - relativedelta(days=1)

        listCurrency = ["EUR","USD","JPY","GBP","SGD","AUD","CNY","CAD","MYR","RUB"]
        date_list = []
        currency_data = {}

        iter = 10

        api_keys = [
        "8fW1FU2clHynfQEFl50XnQiLio3NWMyv7WzWYBig",
        "3FZ5rrqBaQzKQTLwcPtf4Zi3sSwFjE9UvSX4a9wB",
        "SV251ojFa6lUdx8UmayICZ9wTqujY2TDqUC8mEXf",
        "bYbxmf8K1Pp5jtUlZnOKEus1rwVrwTYTbymGQqp7"
        ]

        api_key_index = 0  # Index to track the current API key

        while iter >= 1:
            for currency in listCurrency:
                api_key = api_keys[api_key_index]
                url = f'https://api.freecurrencyapi.com/v1/historical?apikey={api_key}&currencies=IDR&base_currency={currency}&date_from={today - relativedelta(years=iter)}T13%3A51%3A22.659Z&date_to={today - relativedelta(years=iter-1)}T13%3A51%3A22.659Z'
                response = requests.get(url)
                data = response.json()

                for date, value in data['data'].items():
                    if date not in date_list:
                        date_list.append(date)
                    if date not in currency_data:
                        currency_data[date] = {}
                    currency_data[date][currency] = value['IDR']

            print(f'key={api_key} iter={iter}')
            iter -= 1
            api_key_index = (api_key_index + 1) % len(api_keys)  # Move to the next API key

            # Prepare CSV data
            csv_data = []
            csv_data.append(["date"] + listCurrency)
            for date in date_list:
                row = [date] + [currency_data[date][currency] for currency in listCurrency]
                csv_data.append(row)

            # Write CSV file
            with open("currency_data_10_years.csv", "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(csv_data)

            # Print CSV-style output
            for row in csv_data:
                print(",".join(map(str, row)))


