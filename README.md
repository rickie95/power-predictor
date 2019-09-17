# power-predictor - Prophet by Facebook

[From Facebook's github repository](https://github.com/facebook/prophet)

This branch explored the time series reconstruction using a prediction tool developed by Facebook, 
based on a variant of STL decomposition. 

Prophet paper: Sean J. Taylor, Benjamin Letham (2018) Forecasting at scale. The American Statistician 72(1):37-45 

## Installation and setup

**This repository has been tested on Ubuntu 16.04 LTS**. 

Prophet and his dependencies are available on Windows and OSX/MacOs too. Refer to Prophet's github page for detailed 
installing instructions.

Make sure compilers (gcc, g++, build-essential) and Python development tools (python-dev, python3-dev) are installed.
PyStan and Prophet packages need to be compiled and require the appropriate build toolchain.

#### Jupiter notebook
Be sure to have your `dataset.csv` into the `./input` folder, run your server and you're ready to go.

#### Python scripts
**Recommended**: create a dedicate virtual environment for this repository.

Install the required packages using `pip`, there's a specific order to observe while installing modules.
        
        pip install -r requirements.txt
        
Be sure to have your csv files in the `input` folder.

**Scripts available**
+ **prophet_demo.py** takes as input a complete dataset, erases a significative portion of values, then proceeds 
to reconstruct them and plot the result.

+ **crossvalidation.py** scans the `input` folder and collect all the csv inside. Then proceeds to erase a portion of 
values (16/4/2/1 weeks long, see global parameters) and reconstruct that erased values for all datasets. Produces as 
output a report with RMSE, integral of values and standard deviation.

+ **plot.py** an useful pre-baked script to visualize Prophet's output dataframe.
