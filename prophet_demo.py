from script import prepare_dataframe, suppress_stdout_stderr
from fbprophet import Prophet
import datetime
import os


''' A simple demo showing what Prophet can do with a missing values time serie. '''


def main():

    # Prophet requires a Pandas' dataframe with 'y' and 'ds' columns. We need to transform the csv a little.
    dataframe = prepare_dataframe('csv/LHO/KUT_039AFA_LHO.csv')

    # Our data are complete, so we can cut some and see how Prophet reconstructs them.
    # loc: in the first part I've specified the condition requested, in the second I've selected the column
    # to be assigned.
    # "Assign value None in the 'y' column for rows with ds in range 01 Gen 2017 to 26 March 2017"
    dataframe.loc[('2017-01-01 00:00:00' <= dataframe['ds']) & (dataframe['ds'] <= '2017-03-26 00:00:00'), 'y'] = None

    # We also cut 0 and less for power consumption.
    dataframe.loc[(dataframe['y'] <= 0), 'y'] = None

    # Ok, now we create the Prophet model and we fit it with our cutted dataframe.
    model = Prophet(daily_seasonality=20, yearly_seasonality=20)

    # with suppress_stdout_stderr(): if you want no output
    model.fit(dataframe)

    # We need to create a dataframe where store our time serie, reconstructed and provisioned.
    # periods: units of time to be predicted
    # frequency: Time unit (Hours)
    # include_history: includes or not the train data in final dataframe
    future = model.make_future_dataframe(periods=24, freq='H', include_history=True)

    # Finally, we make Prophet do the job.
    forecast = (model.predict(future))

    # Create reconstruction and STL components plots.
    data_fig = model.plot(forecast)
    components_fig = model.plot_components(forecast)

    data_fig.show()
    components_fig.show()

    # Save the results into a csv contained in a dedicated folder
    today = datetime.datetime.now()
    results_dir = "results_" + str(today.year) + "_" + str(today.month) + "_" + str(today.day) + "__" \
                  + str(today.hour) + "_" + str(today.minute) + "_" + str(today.second)
    os.mkdir(results_dir)

    data_fig.savefig(os.path.join(results_dir, "data_plot.pdf"))
    components_fig.savefig(os.path.join(results_dir, "STL_components_plot.pdf"))

    forecast.to_csv(os.path.join(results_dir, "results.csv"), sep=',')


main()
