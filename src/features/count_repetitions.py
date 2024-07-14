import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle(r"C:\Users\ARNAV AGARWAL\OneDrive - Indian Institute of Technology Bombay\Desktop\SS\ML\CodingFitnessTracker\data-science-template\data\interim\01_data_processed.pkl")
#we took the initial data since after that we had removed many data points, for counting reps, lets see which initial features are actually good.

df = df[df["label"]!="rest"]

#lets put squared vars again here, it can be imp
acc_r = df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2
gyr_r = df["gyr_x"]**2 + df["gyr_y"]**2 + df["gyr_z"]**2
df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)



#lets study all exercises separately.
bench_df = df[df["label"]=="bench"]
squat_df = df[df["label"]=="squat"]
row_df = df[df["label"]=="row"]
ohp_df = df[df["label"]=="ohp"]
dead_df = df[df["label"]=="dead"]

plot_df = bench_df
plot_df["set"].unique()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"]==plot_df["set"].unique()[0]]["gyr_r"].plot()
#after seeing graphs, we see that acc data for bench press can be better in terms of counting reps than gyr data.

# Lets configure Lowpassfilter
fs= 1000/200
LowPass = LowPassFilter()
#apply and tweal lowpassfilter

bench_set = bench_df[bench_df["set"]==bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"]==squat_df["set"].unique()[0]]
row_set = row_df[row_df["set"]==row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"]==ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"]==dead_df["set"].unique()[0]]

bench_set["acc_r"].plot()
col = "acc_r"
LowPass.low_pass_filter(bench_set, col=col, sampling_frequency=fs, cutoff_frequency=0.4, order=5)[col+"_lowpass"].plot()
#so after tuning cutoff_frequency, we find that 0.4 is a good value to see all 5 reps in the graph.

# create function to count reps

def count_reps(dataset, cutoff=0.4, order=10, column = "acc_r"):
    data = LowPass.low_pass_filter(dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order)
    indexes = argrelextrema(data[column+"_lowpass"].values,np.greater) #helps identify the peaks, here max ones since we used np.greater
    peaks = data.iloc[indexes]
    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peaks)} Reps")
    plt.show()
    return len(peaks)

count_reps(bench_set, cutoff=0.4, order=10, column = "acc_r") 
#after some tweaking for atleast first set
count_reps(bench_set, cutoff=0.4)
count_reps(squat_set, cutoff=0.35)
count_reps(row_set, cutoff=0.65, column="gyr_x")
count_reps(ohp_set, cutoff=0.35)
count_reps(dead_set, cutoff=0.4)

df["reps"] = df["category"].apply(lambda x:5 if x=="heavy" else 10)
rep_df = df.groupby(["label", "set", "category"])["reps"].mean().reset_index()
rep_df["reps_pred"] = 0

for s in df["set"].unique():
    subset = df[df["set"]==s]
    column = "acc_r"
    cutoff= 0.4
    if subset["label"].iloc[0] == "row":
        column = "gyr_x"
        cutoff = 0.65
    if subset["label"].iloc[0] == "squat":
        cutoff = 0.35
    if subset["label"].iloc[0] == "ohp":
        cutoff = 0.35
        
    reps = count_reps(subset, cutoff=cutoff, column=column, order=10)
    
    rep_df.loc[rep_df["set"]==s, "reps_pred"] = reps #all rows where set is s, we take the reps_pred col and put reps there.
    


#evaluate the results
error = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2) #i.e. on avg, 1 rep is mistaken out of all 85 sets
rep_df.groupby(["label", "category"])[["reps", "reps_pred"]].mean().plot.bar()

 