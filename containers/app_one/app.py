from prometheus_client import start_http_server, Gauge,Histogram, Counter

import random
import time


# Lab 2
g = Gauge('demo_gauge', 'Description of demo gauge')

# New gauges
gauge_metric_1 = Gauge('gauge_metric_1', 'Random variable between 0 and 1')
gauge_metric_2 = Gauge('gauge_metric_2', 'Random variable between 0 and 0.6')

# # New histograms
# histogram_metric_1 = Histogram('histogram_metric_1', 'Random variable between 0 and 1')
# histogram_metric_2 = Histogram('histogram_metric_2', 'Random variable between 0 and 0.6')

# Lab 3
# Define the threshold
threshold = 0.6

# Create the gauges
request_time_train = Gauge('request_time_train', 'Simulated request time for training')
request_time_test = Gauge('request_time_test', 'Simulated request time for testing')

# create histograms for request time
histogram_request_time_train = Histogram('histogram_request_time_train', 'Simulated request time for training')
histogram_request_time_test = Histogram('histogram_request_time_test', 'Simulated request time for testing')


def emit_data(t):
    """Emit fake data"""

    # Set gauge values
    gauge_metric_1.set(t)
    gauge_metric_2.set(t * 0.6)

    # # Set histogram values
    # histogram_metric_1.observe(t)
    # histogram_metric_2.observe(t * 0.6)
    
    # Lab 3
    # Generate a random value between 0 and 1
    value = random.random()

    # Set the value of request_time_train, clipping at the threshold
    if value > threshold:
        request_time_train.set(threshold)
    else:
        request_time_train.set(t)

    # Set the value of request_time_test
    request_time_test.set(t)


    # Set the value of histogram_request_time_train, clipping at the threshold
    if value > threshold:
        histogram_request_time_train.observe(threshold)
    else:
        histogram_request_time_train.observe(t)


    # Set the value of histogram_request_time_test
    histogram_request_time_test.observe(t)

    # Set gauge value
    time.sleep(t)
    g.set(t)


if __name__ == '__main__':
    start_http_server(8000)
    while True:
        emit_data(random.random())



