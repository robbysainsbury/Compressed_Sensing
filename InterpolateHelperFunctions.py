import math
import numpy as np
import pandas as pd
import copy
import random
import matplotlib.pyplot as plt

from CompressedSensingInterpolator import CompresedSensingInterpolator

def getExampleData(numDataPoints=100, proportionMissing=0.5):

    ts = np.linspace(-1, 1, numDataPoints)
    ys = np.sin(ts) + np.cos(2 * ts) + np.sin(ts * 10) + np.cos(ts * 20) + np.cos(ts * 50)
    indices = np.arange(ts.shape[0])
    selectedIndices = np.random.choice(indices, int(numDataPoints * proportionMissing))
    mask = np.in1d(indices, selectedIndices)
    ysMissing = copy.copy(ys)
    ysMissing[mask] = None
    return ys, ysMissing

# making a combined column of the filled in data and the original data, at full resolution
def combine_pressure_row(row):
    if not math.isnan(row['pressure_filled']) and math.isnan(row['pressure_hobo']):
        return row['pressure_filled']
    elif math.isnan(row['pressure_filled']) and not math.isnan(row['pressure_hobo']):
        return  row['pressure_hobo']
    elif not math.isnan(row['pressure_filled']) and not math.isnan(row['pressure_hobo']):
        return  row['pressure_hobo']
    else:
        return np.nan

def down_sample_and_interpolate_once(site_df,all_days,sample_down_to,column):

    ysMissing_df = site_df.iloc[::sample_down_to, :] #sampling only every __ measurement to save on memory
    ysMissing = np.asarray(ysMissing_df[column])

    method="SLSQP" # "BFGS", etc. see the method paramter here -> https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    numBases = int((ysMissing.shape[0]/2) + 1)
    sparse_interpolator = CompresedSensingInterpolator()
    ysComplete = sparse_interpolator.interpolate(ysMissing, numBases=numBases, method=method)

    ysComplete_df = pd.DataFrame(ysComplete, columns=["pressure_filled"])
    joined_df = pd.merge(ysMissing_df.loc[:,'datetime'].reset_index(), ysComplete_df, left_index=True, right_index=True,suffixes=("","_past"))
    sparse_joined_df = all_days.merge(joined_df, on='datetime', how='left')
    sparse_joined_df = sparse_joined_df.loc[:,['datetime','index','pressure_filled']].merge(site_df.loc[:,['datetime','pressure_hobo']], on='datetime', how='left')

    # create a new column and use np.select to assign values to it using our lists as arguments
    sparse_joined_df['pressure_combined'] = sparse_joined_df.apply(combine_pressure_row, axis=1)
    print("NAs per column:")
    print(sparse_joined_df.isna().sum()/sparse_joined_df.shape[0])
    return sparse_joined_df

def down_sample_and_interpolate(site_df,all_days,sample_down_to):
    print(f"Down sampling to every {sample_down_to} measurements")
    site_df = down_sample_and_interpolate_once(site_df,sample_down_to, "pressure_hobo")
    sample_down_to = math.ceil(sample_down_to/2)
    while sample_down_to != 1:
        print(f"Down sampling to every {sample_down_to} measurements")
        site_df = down_sample_and_interpolate_once(site_df,all_days,sample_down_to, "pressure_combined")
        sample_down_to = math.ceil(sample_down_to/2)

    return site_df

def interpolate_sections(sparse_joined_df,sections_number):
    sparse_joined_df_list = np.array_split(sparse_joined_df, sections_number)
    joined_df_list = []
    method="SLSQP" # "BFGS", etc. see the method parameter here -> https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    print("Interpolating Sections")
    for section in sparse_joined_df_list:
        try:
            ysMissing_df = section
            ysMissing = np.asarray(ysMissing_df['pressure_combined'])
            numBases = int((ysMissing.shape[0]/2) + 1)
            sparse_interpolator = CompresedSensingInterpolator()
            ysComplete = sparse_interpolator.interpolate(ysMissing, numBases=numBases, method=method)

            ysComplete_df = pd.DataFrame(ysComplete, columns=["pressure_combined_filled"])
            joined_df = pd.merge(ysMissing_df.reset_index(), ysComplete_df, left_index=True, right_index=True,suffixes=("","_filled"))

            joined_df_list.append(joined_df)

        except:
            print("Section failed to converge, outputting original combined column instead")
            ysMissing_df = section
            ysMissing_df["pressure_combined_filled"] = ysMissing_df["pressure_combined"]
            print(ysMissing_df.isna().sum()/ysMissing_df.shape[0])
            joined_df_list.append(ysMissing_df)

    return pd.concat(joined_df_list)

def get_test_df(site_df):

    test_df = site_df.copy()
    for i in test_df.sample(frac=0.01).index:
        for j in range(random.randint(10, 300)):
            test_df.loc[i+j, "pressure_hobo"] = np.nan

    print(site_df.isna().sum().sum()/site_df.size)
    print(test_df.isna().sum().sum()/test_df.size)

    plt.scatter(x = "datetime", y = "pressure_hobo",data=site_df, label="Actual Data", s = .5, color="#224dab")
    plt.scatter(x = "datetime", y = "pressure_hobo",data=test_df, label="Test Data", s = .5, color="red")

    return test_df