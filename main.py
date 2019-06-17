import pandas as pd
#from fbprophet import Prophet
#from fbprophet.diagnostics import cross_validation, performance_metrics
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime


def prophet(input_file, results_dir):
    # df = pd.read_csv(input_file, decimal=".", delimiter=",")
    df = input_file
    print(df.head())

    # I valori tra 0 e -inf vengono nullati, in modo da poter fare il logaritmo senza noie.
    # Imposto un limite inferiore al modello

    df.loc[(df['y'] <= 0), 'y'] = None
    df['y'] = np.log(df['y'])
    df['floor'] = min(df['y'])

    # Creo un modello e faccio il fit

    m = Prophet(daily_seasonality=20, yearly_seasonality=20)
    print("Fitting model...")
    m.fit(df)

    # Creo un dataframe con le date che mi interessa prevedere. Posso anche scegliere di non fare previsioni in avanti
    # (e ricostruire quindi solo i dati mancanti) non specificando period.

    future = m.make_future_dataframe(periods=(30*24), freq='H', include_history=True)
    future['floor'] = min(df['y'])

    forecast = m.predict(future)
    print("Prediction: ok")

    forecast.to_csv(os.path.join(results_dir, input_file[:-3] + "_prediction.csv"), sep=',')

    # Plotto le componenti STL e i dati interpolati/ricostruiti

    data_fig = m.plot(forecast)
    components_fig = m.plot_components(forecast)

    data_fig.show()
    components_fig.show()

    # Ripristino i dati con un esponenziale

    df['y'] = np.exp(df['y'])
    forecast['yhat'] = np.exp(forecast['yhat'])

    plt.plot(forecast['yhat'][10000:10500], 'r')
    plt.plot(df['y'][10000:10500], 'b')
    plt.show()

    # Eseguo la k-fold cross validation del modello.

    dataframe_cv = cross_validation(m, horizon='1000 hours')
    dataframe_cv.head()

    # Misuro le metriche MSE MASE ecc..

    performance_dataframe = performance_metrics(dataframe_cv)
    performance_dataframe.head()
    performance_dataframe.to_csv(os.path.join(results_dir, input_file[:-3] + '_performance_results.csv'), sep=",")


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

    for i in range(eff_rows.shape[0]):
        timestamps.append(handle_timestamp(eff_rows['LHO.DHH'][i]))
        values.append(eff_rows[col_to_y][i])

    d = {'ds': timestamps, 'y': values}
    dataframe = pd.DataFrame(data=d)

    return dataframe


def main():
    filenames = ['csv/KUT_033CB5_LHO.csv',
                 'csv/KUT_039AFA_LHO.csv',
                 'csv/KUT_050BC8_LHO.csv',
                 'csv/KUT_055F63_LHO.csv',
                 'csv/KUT_0508A9_LHO.csv']
    for file in filenames:
        dataframe = prepare_dataframe(file)
        plt.plot(dataframe['y'])
        plt.show()
        today = datetime.datetime.now()
        results_dir = "results_" + str(today.year) + "_" + str(today.month) + "_" + str(today.day) + "__" \
                      + str(today.hour) + "_" + str(today.minute)
        os.mkdir(results_dir)
        #prophet(dataframe, results_dir)

main()