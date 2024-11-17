# CS5344_Group10

## Environment profile

environment.txt and environment.yml

## dataset

input_attributes.csv : input feature dataset

labels.csv : target label dataset

## timeseries regression

baseline.py : baseline model for timeseries regression

timeseries.py : define and test four models, which are LightGBM model based on time series, XGBoost model based on time series, two stacked models based on time series

## feature selection

SeleckKbest.py : the baseline of feature selection

featureselection.py : Simulated annealing feature selection algorithm with initial weights

print_featureselection.py : Used to draw the printed result of the previous feature selection into an intuitive image

## other

normalization.py : Used to verify the performance of different normalized strategies

chainregression.py : Chain training framework for solving multi-output regression

All the experimental results were presented in five forms: mse, mae, rmse, mape and r2.
