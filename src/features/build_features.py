import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
df = pd.read_pickle(r"C:\Users\ARNAV AGARWAL\OneDrive - Indian Institute of Technology Bombay\Desktop\SS\ML\CodingFitnessTracker\data-science-template\data\interim\02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight") 
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df.info()

subset = df[df["set"]==35]["gyr_z"].plot()
df["acc_x"] = df["acc_x"].interpolate()
#Dealing with missing values
# We can use interpolate fn for every col
for col in predictor_columns:
    df[col] = df[col].interpolate()
    
df.info()



#Calculating avg time for set( will be used in butterworth filter)

df.query("participant =='A'").query("label=='ohp'").query("category=='heavy'").query("set == 12")["acc_y"].plot()
# so we can see 5 reps for this set, graph goes up and downa and up and down. But  in between there are small peaks like noise which we dont want. So lets fix it.

duration = df[df["set"]==12].index[-1] - df[df["set"]==12].index[0]
duration.seconds

for s in df["set"].unique():
    start = df[df["set"]==s].index[0]
    stop = df[df["set"]==s].index[-1]
    duration = stop - start
    df.loc[df["set"]==s, "duration"] = duration.seconds 
    
duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0]/5 #5 since we have 5 reps in heavy
duration_df.iloc[1]/10


#Using Butterworth filter to remove high freq noise from the dataset

df_lowpass= df.copy()
LowPass = LowPassFilter()
fs = 1000/200 #(1000 is 1 second, and 200 ms is the difference between each record in data. We are getting 5 instances per second)
cutoff = 1.3
df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)


subset = df_lowpass[df_lowpass["set"]==12]
print(subset['label'][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True),label = "raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True),label = "butterworth filter")
ax[0].legend(loc= "upper center", bbox_to_anchor=(0.5, 1.15), shadow=True, fancybox=True)
ax[1].legend(loc= "upper center", bbox_to_anchor=(0.5, 1.15), shadow=True, fancybox=True)
#lower the cutoff, smoother the data

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
    
    


#Using PCA to reduce the dimensionality of the dataset necessary
df_pca = df_lowpass.copy() #to reset the df if needed
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
# lets decide the potential number of components to keep by elbow method

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("Number of components")
plt.ylabel("Explained variance")
plt.show()
# clearly elbow at 3

df_pca= PCA.apply_pca(df_pca, predictor_columns, 3) #we basically summarized the 6 features into 3 pca components
#lets not overwrite original features now

subset = df_pca[df_pca["set"]==12]
subset[["pca_1", "pca_2", "pca_3"]].plot()



#To further exploit the data, the scalar magnitudes r of the accelerometer and gyroscope were
# calculated. r is the scalar magnitude of the three combined data points: x, y, and z. The
# advantage of using r versus any particular data direction is that it is impartial to device
# orientation and can handle dynamic re-orientations. r is calculated by:

df_squared = df_pca.copy()
acc_r= df_squared["acc_x"]**2 + df_squared["acc_y"]**2 + df_squared["acc_z"]**2
gyr_r= df_squared["gyr_x"]**2 + df_squared["gyr_y"]**2 + df_squared["gyr_z"]**2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_pca["set"]==12]
subset[["acc_r", "gyr_r"]].plot(subplots=True)




#Temporal Abstraction
df_temporal = df_squared.copy()

NumAbs = NumericalAbstraction()
predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

ws = int(1000/200) #window size of 5, i.e. inlcuding itself and 4 rows behind, it will take mean and add it to the row.
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean") # col has to be in list format obv
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std") # col has to be in list format obv

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"]==s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)
    
df_temporal= pd.concat(df_temporal_list)


subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()




#Frequency features
df_freq = df_temporal.copy().reset_index()
freqAbs = FourierTransformation()

fs = int(1000/200)
ws = int(2800/200) #avg len of a rep

df_freq = freqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

subset = df_freq[df_freq["set"]==15]
subset[["acc_y"]].plot()
subset[[
    "acc_y_max_freq",
    "acc_y_freq_weighted",
    "acc_y_pse",
    "acc_y_freq_1.429_Hz_ws_14",
    "acc_y_freq_2.5_Hz_ws_14",
]].plot()

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformation to set {s}...")
    subset = df_freq[df_freq["set"]==s].reset_index(drop=True).copy()
    subset = freqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)
    
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True) 



#dealing with overlapping columns
# In the new columns we added, values in rows are highly corellated since rolling window. This could lead to overfitting
#first finish the missing values
df_freq = df_freq.dropna()

# Mostly, 50% overlapping between windows is permitted. For small dataset, allowance can go upto 80 or 90%
df_freq = df_freq.iloc[::2] #i.e. every second row. We reduce corellation.




#Clustering
# Lets see how groups can it come up with

df_cluster = df_freq.copy()
cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2,10)
inertias = []
for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init = 20, random_state=0)  
    cluster_labels = kmeans.fit_predict(subset) #each row will be assigned to a cluster
    inertias.append(kmeans.inertia_)
    
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias, "o-")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()
# THUS K = 5

kmeans = KMeans(n_clusters=5, n_init = 20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot( projection='3d')
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"]==c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label= c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot( projection='3d')
for c in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"]==c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label= c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()




#export
df_cluster.to_pickle(r"C:\Users\ARNAV AGARWAL\OneDrive - Indian Institute of Technology Bombay\Desktop\SS\ML\CodingFitnessTracker\data-science-template\data\interim\03_data_features.pkl") 