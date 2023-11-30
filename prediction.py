# =========== Imports =================
import os
import random
import time
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import xarray as xr
import zarr as zr
import joblib as jb
import time
import pandas as pd
import h5py
import dask.array as da
import math
import glob
import joblib
from joblib import Parallel, delayed

from numpy.lib.stride_tricks import sliding_window_view
import yaml
import sys

# =========== System arguments =================
'''
model_path: Path to the saved model
challenge: name of the challenge, {core, nowcasting, transfer_learning}
len_seq_predict: number of prediction sequence time instance
year: year
region: region name
inputs_root: root directory of input files, possibly data in the starter kit
target_folder: target folder to save model outputs
'''
arguments = sys.argv[1:]
model_path = arguments[0]
challenge = arguments[1]
len_seq_predict = np.int(arguments[2])
year = arguments[3]
region = arguments[4]
inputs_root = arguments[5]
target_folder = arguments[6]

data_split_file_path = 'timestamps_and_splits_stage2.csv'
def standardise_time_strings(time):
    if len(time) < 6:
        time = time.rjust(6, "0")
    else:
        return time
    return time
def get_hours_minutes_seconds(date_time_str):
    h = date_time_str[9:11]
    m = date_time_str[11:13]
    s = date_time_str[13:15]
    return int(h), int(m), int(s)
def get_day_month_year(date_time_str):
    yy = date_time_str[0:4]
    mm = date_time_str[4:6]
    dd = date_time_str[6:8]
    return int(dd), int(mm), int(yy)
def load_timestamps(
    path, types={"time": "str", "date_str": "str", "split_type": "str"}
):
    """load timestamps from a from csv file

    Args:
        path (String): path to the csv file
        types (dict, optional): types to cast columns of dataframe to. Defaults to {'time': 'str', 'date_str': 'str', 'split_type': 'str'}.

    Returns:
        Dataframe: dataframe with timestamps
    """

    df = pd.read_csv(path, index_col=False, dtype=types)

    df.sort_values(by=["date_str", "time"], inplace=True)

    # to datetime type
    df["date"] = pd.to_datetime(df["date"])

    # convert times to strings
    df["time"] = df["time"].astype(str)
    df["time"] = df["time"].apply(standardise_time_strings)

    df["date_str"] = df["date_str"].astype(str)
    # create date_time string
    df["date_time_str"] = df["date_str"] + "T" + df["time"]

    return df

# ============== Testing indices ==================
def get_test_heldout_idxs(df, len_seq_in, data_split, region, years):
    """get sample idxs for test data split

    Args:
        df (DataFrame): _description_
        len_seq_in (int): length of input sequence
        len_seq_predict (int): length of prediction sequence
        data_split (String): data split to load sample idxs for
        region (String): region to create samples for

    Returns:
        list: list of sample idxs for test data split
    """
    idxs = []

    split_type = f"{data_split}_in"
    df = df[df["split_type"] == split_type]
    df = df[df["all_vars"] == 1]
    dfs= pd.DataFrame(columns = df.columns);
    for year in years:
        dfs=pd.concat([dfs,df[df['date'].dt.year==int(year)]]);

    df=dfs;


    for start_index in range(0, df.shape[0], len_seq_in):
        in_seq = [start_index + i for i in range(len_seq_in)]
        test_seq = [in_seq]
        idxs.append(test_seq)
    return idxs

# ================ Load model =========================
dim_x = 252
dim_y = 252
len_seq_in = 4
HRIT_maxs = np.array([[  1.13784206, 336.21588135, 326.39144897, 301.00662231,
        338.03747559, 338.87939453, 300.855896  ,   1.0613029 ,
          1.24437237, 262.51171875, 290.62542725]]).reshape(1,-1)


model = tf.keras.models.load_model(model_path, compile=False)
df = load_timestamps(data_split_file_path)
idx = get_test_heldout_idxs(df, len_seq_in, 'test', region, [year])

inputs_path = f'{inputs_root}/{year}/HRIT/{region}.test.reflbt0.ns.h5'
f = h5py.File(inputs_path, "r")
x_test = f[list(f.keys())[0]]

y_pred = np.zeros((len(idx),len_seq_predict, dim_x, dim_y),dtype=np.float16)
for i in range(len(idx)):
    inp = np.transpose(x_test[idx[i][0][0]:idx[i][0][-1]+1],(0,2,3,1))
    inp = inp/HRIT_maxs
    inp = np.split(inp,inp.shape[0],axis=0)
    inp = np.concatenate(inp,axis=3)
    
    pred = model.predict(inp,batch_size=1)
    pred = np.power(10,(pred*np.log10(128+1)))-1
    pred = pred.astype(np.float16)
    pred = np.transpose(pred,(0,3,1,2))
    y_pred[i] = pred

os.system(f'mkdir -p {target_folder}/{challenge}/{year}')
filename = f"{target_folder}/{challenge}/{year}/{region}.pred.h5"
if os.path.exists(filename):
    os.remove(filename)
h5f = h5py.File(filename, "w")
h5f.create_dataset("submission", data=y_pred, compression='gzip', compression_opts=9)
h5f.close()