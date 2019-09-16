from multiprocessing import Pool
from fbprophet import Prophet
from os import path
import numpy as np
from prophet_utils import *
import glob

'''
    Author: Riccardo Malavolti
    This script does a cross-validation test on a set of time series, calculating the number of splits given one or
    more split lengths in weeks.
    
    In order to be tested with Prophet, time series must be contained in a pandas dataframe with columns "ts" and 
    "y" (respectively timestamp and value of the serie): default granularity is hourly, but you can change if your 
    data are sampled differently.
'''

# GLOBAL PARAMETERS

threads = 4                 # Specifies the number of jobs to run in parallel.
input_dir = "csv/LHO/"      # Path to input files. Can be absolute.
week_split = [16, 4, 2, 1]  # Sizes of splits, specified in weeks.
dd_seasonality = 20         # Order of approximation for daily STL component.
yy_seasonality = 20         # Order of approximation for yearly STL component.
granularity = 'H'


def worker(passthrough):
    dataframe, split_numb, split_total, fract, filename = passthrough

    # A copy is needed, since it's going to null all the values in the planned period of time.
    # It has to be a distinct object.
    df = dataframe.copy(deep=True)
    df.loc[(fract * split_numb):(fract * (split_numb + 1)), 'y'] = None

    m = Prophet(daily_seasonality=dd_seasonality, yearly_seasonality=yy_seasonality)
    with suppress_stdout_stderr():
        m.fit(df)

    # The future dataframe doesn't need to have a period specified, indeed nothing can be inserted, resulting in a
    # pure reconstruction of missing values.

    future = m.make_future_dataframe(periods=1, freq=granularity, include_history=True)
    forecast = (m.predict(future))
    end = fract * (split_numb + 1)

    if fract * (split_numb + 1) >= dataframe.shape[0]:
        end = dataframe.shape[0] - 1

    original = np.copy(dataframe.loc[fract * split_numb:end, 'y'])
    predicted = np.copy(forecast.loc[fract * split_numb:end, 'yhat'])

    numeric_integral_difference = np.nansum(original) - np.nansum(predicted)
    squared_errors = np.power((predicted - original), 2)
    mse_sum = np.nansum(squared_errors)
    mse = mse_sum / original.shape[0]
    std_dev = np.sqrt(np.power(np.nansum(np.subtract(predicted, mse)), 2) / original.shape[0])

    print("JOB " + filename + ": " + str(split_numb + 1) + "/" + str(split_total) + " MSE:" + str(mse))
    return filename, split_numb + 1, mse, numeric_integral_difference, std_dev


def prophet(input_file, splits):
    # Creo un pool di thread
    fraction = input_file.shape[0] / splits
    pool = Pool(processes=threads)

    input_list = []
    for i in range(splits):
        input_list.append((input_file, i, splits, fraction, input_file.filename))

    return pool.map(worker, input_list)  # returns a list of tuples with results


def crossvalidation(dataframe, results_dir, weeks):
    splits = calculate_splits(dataframe['ds'], weeks)
    print("WEEK(S): {} SPLITS: {}".format(weeks, splits))
    results = prophet(dataframe, splits)
    # Writes result in a csv file
    with open((path.join(results_dir, "crossvalidation_results.csv")), "a+") as f:
        line = "{0};{1};{5};{6};{2};{3};{4}\n"
        for res in results:
            f.write(line.format(*res, splits, weeks))


def calculate_splits(timestamps, weeks):
    max_ts = max(timestamps)
    min_ts = min(timestamps)
    n_dd = max_ts - min_ts
    splits = int(n_dd.days / (weeks * 7))
    return splits


def main():
    weeks = week_split

    results_dir = create_results_dir()

    filenames = glob.glob(input_dir + "*.csv")  # Finds all csv files in input dir

    for file in filenames:
        print("\n### " + file + " ### \n")
        dataframe = prepare_dataframe(file)
        dataframe.loc[(dataframe['y'] <= 0), 'y'] = None
        for w in weeks:
            crossvalidation(dataframe, results_dir, w)


main()
