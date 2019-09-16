import pandas as pd
import datetime
import os


def create_results_dir():
    today = datetime.datetime.now()
    results_dir = "results_" + str(today.year) + "_" + str(today.month) + "_" + str(today.day) + "__" \
                  + str(today.hour) + "_" + str(today.minute) + "_" + str(today.second)
    os.mkdir(results_dir)
    return results_dir


def prepare_dataframe(filename, col_to_y='LHO.W1'):
    """
        Takes as input a csv file, creates a pandas dataframe with [ ds , y] columns, where:
        - ds is a timestamp YYYY-MM-DD HH:MM
        - y is a float64 for col_to_y column's values
    """

    def handle_timestamp(timestamp):
        """
            :param timestamp: a string in the YYYYmmDDHHMM format
            :return: a datetime obj
        """
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

    # Only "effective" marked rows.
    eff_rows = df.loc[df['LHO.FCP'] == 'E']

    for i in eff_rows.index:
        timestamps.append(handle_timestamp(eff_rows['LHO.DHH'][i]))
        values.append(eff_rows[col_to_y][i])

    d = {'ds': timestamps, 'y': values}
    dataframe = pd.DataFrame(data=d)
    dataframe.filename = os.path.splitext(os.path.split(filename)[1])[0]

    return dataframe


class suppress_stdout_stderr(object):
    """
        A context manager for doing a "deep suppression" of stdout and stderr in
        Python, i.e. will suppress all print, even if the print originates in a
        compiled C/Fortran sub-function.

        This will not suppress raised exceptions, since exceptions are printed
        to stderr just before a script exits, and after the context manager has
        exited.
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
