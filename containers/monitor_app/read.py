# Using the sample Notebook from class, prepare a Python application (not Notebook) that
# reads the training data metric for a period of time, builds a Prophet model from it, and
# evaluates the model against the test metric. Simply print the anomalies detected to the
# console log.
# Important hints:
# • Be careful to not overlap your train and test times, e.g. pull the training metric for the last 5 min, wait a
# minute, then pull the test metric just for the previous minute. Then evaluate the model against the past
# minute’s test metric for anomalies.
# • You’ll likely find that the model drifts rapidly even after just a few minutes and you start getting “unrea-
# sonable” anomalies. Update your application to repeat the model training cycle (fetch data, train model)
# on each 60-second iteration. As always, be careful to not overlap train and test data as above

import pandas as pd
import time
from prometheus_client.parser import text_string_to_metric_families
import requests
from datetime import datetime
from prophet import Prophet
from prometheus_client import start_http_server,Gauge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import argparse


def fetch_data(metric_name,time):
    query = f"http://prometheus:9090/api/v1/query?query={metric_name}[{int(time/60)}m]"
    # print fetch query
    # print(query)
    # query = f"http://localhost:9090/api/v1/query_range?query={metric_name}&start={start_time}&end={end_time}&step=1m"
    response = requests.get(query)
    if response.status_code != 200:
        print(f"Error: Request failed with status code {response.status_code}")
        return []  

    try:
        data = response.json()['data']['result'][0]['values']
        return data
    except (KeyError, ValueError):
        print("Error: Invalid JSON data received from Prometheus.")
        return []

def prepare_data(data):
    df = pd.DataFrame(data, columns=['ds', 'y'])
    df['ds'] = pd.to_datetime(df['ds'],unit='s')
    df['y'] = pd.to_numeric(df['y'])
    # Reset the index to sequential integers starting from 1
    df.reset_index(drop=True, inplace=True)
    return df


def detect_anomalies(model, test_df):
    forecast = model.predict(test_df)
    # Merge actual and predicted values
    performance = pd.merge(test_df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    # Create an anomaly indicator
    performance['anomaly'] = performance.apply(lambda rows: 1 if ((float(rows.y)<rows.yhat_lower)|(float(rows.y)>rows.yhat_upper)) else 0, axis = 1)
    # Take a look at the anomalies
    anomalies = performance[performance['anomaly']==1].sort_values(by='ds')
    # forecast['anomaly'] = ((forecast['y'] > forecast['yhat_upper']) | (forecast['y'] < forecast['yhat_lower'])).astype(int)
    # anomalies = forecast[forecast['anomaly'] == 1]
    
    return anomalies, performance


# Create a Prometheus counter for anomaly count
anomaly_count = Gauge('anomaly_count', 'Anomaly count in the monitoring application')
# Create Prometheus Gauges for MAE and MAPE
mae_gauge = Gauge('mae', 'Mean Absolute Error of the model')
mape_gauge = Gauge('mape', 'Mean Absolute Percentage Error of the model')


# # Define start-up parameters
# TRAINING_DURATION = 300  # Number of seconds of training time to extract from Prometheus
# FORECAST_DURATION = 60  # Number of seconds to forecast into the future
# FORECAST_ITERATIONS = 10  # Number of times a forecast should be made before retraining the model

if __name__ == "__main__":
   
    # Start Prometheus client
    start_http_server(8003)

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t') #training time
    parser.add_argument('-T') #test time 
    parser.add_argument('-f') #forecast count
    parser.add_argument('-w') #wait time
    args = parser.parse_args()

    TRAINING_DURATION = int(args.t)
    FORECAST_DURATION = int(args.T)
    FORECAST_ITERATIONS = int(args.f)
    WAIT_TIME = int(args.w)
    
    
    # sleep training time
    time.sleep(WAIT_TIME)


    while True:
        # Dataframe to store metrics for each forecast step
        metrics_df = pd.DataFrame(columns=['current_time', 'anomaly_count', 'MAE', 'MAPE'],index=np.arange(FORECAST_ITERATIONS))

        # Fetch and prepare training data
        train_data = fetch_data('request_time_train', TRAINING_DURATION)
        train_df = prepare_data(train_data)

       
        for _ in range(FORECAST_ITERATIONS):         
           

            # Train the model
            model = Prophet(interval_width=0.99, yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
            model.fit(train_df)

            # Fetch and prepare test data
            test_data = fetch_data('request_time_test', FORECAST_DURATION)
            test_df = prepare_data(test_data)
            
            # Detect anomalies
            anomalies,performance = detect_anomalies(model, test_df)

            # if not anomalies.empty:
            print("Detected Anomalies:")
            print(anomalies)

            # Set the anomaly gauge to sum of anomalies detected over a period
            anomaly_count.set(len(anomalies))

            # Calculate MAE and MAPE
            y_true = test_df['y']
            y_pred = performance['yhat']
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)

            # print
            print("The MAE for the model is " + str(mae))
            print("The MAPE for the model is " + str(mape))


            # Set the MAE and MAPE gauges
            mae_gauge.set(mae)
            mape_gauge.set(mape)

            # get anomaly_count for each forecast step where anomaly=1
            an_cnt= performance[performance['anomaly'] == 1]['anomaly'].count()


            # Add metrics to the metrics dataframe for each forecast step
            # do it for each forecast step
            metrics_df.loc[_] = [datetime.now(), an_cnt, mae, mape]
            # print(metrics_df)       
            
            
            time.sleep(FORECAST_DURATION)

        # Print dataframe when application is exiting
        print(metrics_df)

            