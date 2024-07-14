import pandas as pd
from glob import glob
import numpy as np
import os
single_file_acc = pd.read_csv(r"C:\Users\ARNAV AGARWAL\OneDrive - Indian Institute of Technology Bombay\Desktop\SS\ML\CodingFitnessTracker\data-science-template\data\raw\MetaMotion\A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
single_file_gyr = pd.read_csv(r"C:\Users\ARNAV AGARWAL\OneDrive - Indian Institute of Technology Bombay\Desktop\SS\ML\CodingFitnessTracker\data-science-template\data\raw\MetaMotion\A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

#lets list all the files(data) we are going to use and we do this by using glob
files = glob(r"C:\Users\ARNAV AGARWAL\OneDrive - Indian Institute of Technology Bombay\Desktop\SS\ML\CodingFitnessTracker\data-science-template\data\raw\MetaMotion\*.csv") #i.e.list only all the csv files in that path
# files1 = [os.path.normpath(file).replace("\\","/") for file in files]
len(files)
files[0]

# Lets get the pieces of the string to create a dataframe
data_path = "template\\data\\raw\\MetaMotion\\"
f = files[0]

participant = f.split("-")[3].replace(data_path, "")
label = f.split("-")[4]
category = f.split("-")[5].rstrip("123").rstrip("_MetaWear_2019") #rstrip removes any of the char in the end that it finds matching to this "2"

df = pd.read_csv(f)

#lets add columns : participant, label, category to the dataframe
df["participant"] = participant
df["label"] = label
df["category"] = category


#lets apply to all the files through a function
acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    participant = f.split("-")[3].replace(data_path, "")
    label = f.split("-")[4]
    category = f.split("-")[5].rstrip("123").rstrip("_MetaWear_2019")
    
    df = pd.read_csv(f)
    
    df["participant"] = participant
    df["label"] = label
    df["category"] = category
    
    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])
        
    elif "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])
        

# gyr_set has higher amount of data since it was measuring at a higher frequency than the accelerometer

#WORKING WITH DATETIMES
acc_df.info()

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
# acc_df["time (01:00)"]
# There is a difference of one hr since its a result of differnece between UTC time and CET winter time.
# pd.to_datetime(acc_df["time (01:00)"]) #changes dtype to datetime64[ms] so that we can use its attributes
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

del acc_df["epoch (ms)"]
del gyr_df["epoch (ms)"]
del acc_df["time (01:00)"]
del gyr_df["time (01:00)"]
del acc_df["elapsed (s)"]
del gyr_df["elapsed (s)"]


#Lets turn everything into function
files = glob(r"C:\Users\ARNAV AGARWAL\OneDrive - Indian Institute of Technology Bombay\Desktop\SS\ML\CodingFitnessTracker\data-science-template\data\raw\MetaMotion\*.csv")

def read_data_from_files(files):
    
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("-")[3].replace(data_path, "")
        label = f.split("-")[4]
        category = f.split("-")[5].rstrip("123").rstrip("_MetaWear_2019")
        
        df = pd.read_csv(f)
        
        df["participant"] = participant
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
            
        elif "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])
            
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del gyr_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del gyr_df["time (01:00)"]
    del acc_df["elapsed (s)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df, gyr_df

acc_df, gyr_df = read_data_from_files(files)

#merging datasets

data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)  #reference is always the index i.e. epoch(ms).
# You can see almost all rows has something missing either in acc or gyr, since gyr has higher freq, and prob of matching at the exact ms is very low. We get approx 1000 rows out ot 70k rows which has all column data
data_merged.columns = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "participant", "label", "category", "set"]

#Thus we resample data

#Accelerometer: 12.500 Hz
#Gyroscope: 25.000 Hz
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "label": "last",
    "category": "last",
    "participant":"last",
    "set":"last"
}
data_merged[:1000].resample(rule="200ms").apply(sampling)  

days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat((df.resample(rule="200ms").apply(sampling).dropna() for df in days))

data_resampled["set"] = data_resampled["set"].astype(int)
data_resampled.info()

#export data as pkl file
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl") 
# pkl is good since file size of pkl is small and we dont need any conversions again and again