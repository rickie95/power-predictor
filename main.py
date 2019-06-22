from fbprophet import Prophet
from multiprocessing import Process
from os import path
import pandas as pd
import numpy as np
import datetime
import os


def prophet(input_file, results_dir, log, splits):
    # df = pd.read_csv(input_file, decimal=".", delimiter=",")
    df = input_file
    print(df.head())

    # I valori tra 0 e -inf vengono nullati, in modo da poter fare il logaritmo senza noie.
    # Imposto un limite inferiore al modello

    df.loc[(df['y'] <= 0), 'y'] = None
    if log:
        df['y'] = np.log(df['y'])
    df['floor'] = min(df['y'])

    mse_values = []

    # Creo un modello e faccio il fit

    k = splits
    for i in range(k):

        df_copia = df.copy(deep=True)
        fraction = df.shape[0]/k
        df_copia.loc[(fraction*i):(fraction*(i+1)), 'y'] = None

        m = Prophet(daily_seasonality=20, yearly_seasonality=20)
        print("Fitting model...(" + str(i+1) + "/"+str(k)+")")
        m.fit(df)

        # Creo un dataframe con le date che mi interessa prevedere. Posso anche scegliere di non
        # fare previsioni in avanti (e ricostruire quindi solo i dati mancanti) non specificando period.

        future = m.make_future_dataframe(periods=(24), freq='H', include_history=True)
        future['floor'] = min(df['y'])

        forecast = (m.predict(future))

        end = fraction*(i+1)
        if fraction*(i+1) >= df.shape[0]:
            end = df.shape[0]-1

        original = np.copy(df.loc[fraction*i:end, 'y'])
        predicted = np.copy(forecast.loc[fraction*i:end, 'yhat'])

        if log:
            original = np.exp(original)
            predicted = np.exp(predicted)

        numeric_integral_difference = np.nansum(original) - np.nansum(predicted)
        squared_errors = np.power((predicted - original), 2)
        mse_sum = np.nansum(squared_errors)
        mse = mse_sum / original.shape[0]

        print("MSE run "+str(i+1) + "/"+str(k)+": "+str(mse))
        mse_values.append((input_file.filename, i+1, mse, log, numeric_integral_difference))

        print("Prediction: ok (" + str(i+1) + "/"+str(k)+")")

    #forecast.to_csv(os.path.join(results_dir, input_file.filename + "_mse.csv"), sep=',')

    # Plotto le componenti STL e i dati interpolati/ricostruiti

    #data_fig = m.plot(forecast)
    #components_fig = m.plot_components(forecast)

    #data_fig.show()
    #components_fig.show()

    # Ripristino i dati con un esponenziale



    #plt.plot(forecast['yhat'][10000:10500], 'r')
    #plt.plot(df['y'][10000:10500], 'b')
    #plt.show()

    return mse_values


def prepare_dataframe(filename, col_to_y='LHO.W1'):
    # Prende un fie csv e prepara un dataframe con struttura [ ds , y] dove:
    #    - ds è un timestamp YYYY-MM-DD HH:MM
    #    - y è un float64 con i valori su cui lavorare

    def handle_timestamp(timestamp):
        ''' :param timestamp: Una stringa YYYYmmDDHHMM
        :return: un datetime obj
        '''
        timestamp = str(timestamp) + '00'

        if timestamp[8:10] == '24':
            timestamp_list = list(timestamp)
            timestamp_list[8:10] = '23'
            timestamp = "".join(timestamp_list)
            timestamp = datetime.datetime.strptime(timestamp, '%Y%m%d%H%M%S') + datetime.timedelta(hours=1)
        else:
            timestamp = datetime.datetime.strptime(timestamp, '%Y%m%d%H%M%S')

        return timestamp

    df = pd.read_csv(filename, decimal=".", delimiter=",")
    timestamps = []
    values = []

    # estraggo solo le righe effettive
    eff_rows = df.loc[df['LHO.FCP'] == 'E']

    for i in eff_rows.index:
        timestamps.append(handle_timestamp(eff_rows['LHO.DHH'][i]))
        values.append(eff_rows[col_to_y][i])

    d = {'ds': timestamps, 'y': values}
    dataframe = pd.DataFrame(data=d)

    return dataframe


def crossvalidate(dataframe, results_dir, number_of_splits):
    mse_values = []
    mse_values.extend(prophet(dataframe, results_dir, False, number_of_splits))
    mse_values.extend(prophet(dataframe, results_dir, True, number_of_splits))
    with open((path.join(results_dir, dataframe.filename + "_results.csv")), "w") as f:
        for elem in mse_values:
            f.write((str(elem[0]) + ";" + str(elem[1]) + "," + str(elem[2]) + ";" + str(elem[4]) + ";" +
                     str(elem[3]) + "\n"))


def main():
    today = datetime.datetime.now()
    results_dir = "results_" + str(today.year) + "_" + str(today.month) + "_" + str(today.day) + "__" \
                  + str(today.hour) + "_" + str(today.minute) + "_" + str(today.second)
    os.mkdir(results_dir)
    filenames = ['KUT_033CB5_LHO.csv',
                 'KUT_039AFA_LHO.csv',
                 'KUT_050BC8_LHO.csv',
                 'KUT_055F63_LHO.csv',
                 'KUT_0508A9_LHO.csv']
    threads = []
    delta = 7*4

    for file in filenames:
        print("\n### " + file + " ### \n")
        dataframe = prepare_dataframe('csv/LHO/' + file)
        dataframe.filename = file[:-4]
        max_ts = max(dataframe['ds'])
        min_ts = min(dataframe['ds'])
        n_dd = max_ts - min_ts
        splits = int(n_dd.days / delta)
        print("SPLITS: "+str(splits))
        #plt.plot(dataframe['y'])
        #plt.xlabel(file)
        #plt.show()

        #lancia thread
        p = Process(target=crossvalidate, args=(dataframe, results_dir, splits))
        p.start()
        threads.append(p)

    for t in threads:
        t.join()


main()