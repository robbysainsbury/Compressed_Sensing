# import necessary libraries
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
from functools import reduce
import matplotlib.pyplot as plt


def round_dt(dt):
    dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    delta = timedelta(minutes=15)
    return datetime.min + round((dt - datetime.min) / delta) * delta

site_names_list = ['BEN', 'BLI', 'BSL', 'CLE', 'CRB', 'DAI', 'DFF', 'DFL', 'DFM', 'DFU', 'HCL', 'HCN', 'HCS', 'IND', 'LAK', 'LDF', 'MIT', 'NEB', 'PBC', 'SBL', 'SFL', 'SHE', 'SHE_', 'SOL', 'SOL_', 'STR', 'TCU', 'TIE', 'WAN']
  
# use glob to get all the csv files 
# in the folder
path = "../cleaned_pressure_data/"
csv_files = glob.glob(os.path.join(path, "*.csv"))

site_DF_list = []
# loop over the list of csv files
for i in range(len(csv_files)):

    f = csv_files[i]
    site_name = site_names_list[i]
    col_names = ['datetime', site_name] 

    # read the csv file
    df = pd.read_csv(f, usecols=range(2,4),  skiprows=[0], names = col_names)

    #rounding each datetime to the nearest 15 minutes
    df['datetime'] = df['datetime'].apply(round_dt)

    #adding to the list
    site_DF_list.append(df) 

#merging into one dataframe
pressDF = reduce(lambda  left,right: pd.merge(left,right,on=['datetime'],
                                            how='outer'), site_DF_list)

#creating average pressure column 
pressDF["Average"] = pressDF.iloc[:,1:].mean(axis=1, skipna=True)

#exporting csv
pressDF.to_csv("all_pressure.csv")

