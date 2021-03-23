# Covid-19
This is a folder containing scripts for Covid 19 cases prediction.  

## Data
Data File : time_series_covid19_confirmed_US.csv (Original Source : https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv). 

## Set up
Requirements : 
  * Python > 3.6 version 
  * Tensorflow, keras 
Command : python3.6 covid_forecast.py

## Descriptions 
* covid_forecast.py - For state of Virginia, change the state name in the file to run for any other state.
* covid_forecast_ind_parallel.py - Running for each county individually for Virginia State counties. change the state name in the file to run for any other state.
* covid_forecast_all_states.py - To run for all the 50+ Provincial states individually. 

* utils.py, ML_models.py, ML_GP.py, POD.py : Files containing utility functions and time series models. 


               
               
