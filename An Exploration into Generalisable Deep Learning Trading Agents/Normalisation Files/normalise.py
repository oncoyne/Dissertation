import glob
import os
import pickle
import csv
import pandas as pd
import numpy as np
import dask.dataframe as dd

#Ingest Data
headings = ["Time", "Type", "Limit", "Spread", "Midprice", "Microprice", "Best Bid", "Best Ask", "timedelta", "Imbalance", "Total Quotes", "Equilibrium", "Smith's Alpha", "Price"]
feature_headings = ["Time", "Type", "Limit", "Spread", "Midprice", "Microprice", "Best Bid", "Best Ask", "timedelta", "Imbalance", "Total Quotes", "Equilibrium", "Smith's Alpha"]
df = dd.read_csv('D:/Thesis/Dataset/PTransactions\*.csv',names=headings)

#Remove collumns for Meades dataset
#df = df.drop(columns = ["Time", "Best Ask", "timedelta", "Imbalance", "Total Quotes", "Equilibrium", "Smith's Alpha" ])

#Normalise the data
df_normalised = (df-df.min())/(df.max()-df.min())

#set_mean = df.mean().compute()
#set_stdev = df.std().compute()
#set_mean = df_normalised.mean().compute()
#print(set_mean)
#print(set_stdev)
#print(df_normalised.head(20))

#Store normalised data within HDF file format
print(df_normalised.compute())
df_normalised.to_hdf("D:/Thesis/Dataset/PRZI.hdf","/data")

#Compute and store the values required to denormalise network output
min_vals = df.min().compute()
max_vals = df.max().compute()

min_max_dict = {"Min":min_vals,"Max":max_vals}
min_max = pd.DataFrame.from_dict(min_max_dict)
min_max.to_csv("D:/Thesis/Dataset/PRZI_min_max.csv")

print("Done")
