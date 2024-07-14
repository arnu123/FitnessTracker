import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor

df = pd.read_pickle(r"C:\Users\ARNAV AGARWAL\OneDrive - Indian Institute of Technology Bombay\Desktop\SS\ML\CodingFitnessTracker\data-science-template\data\interim\01_data_processed.pkl")

#Mostly numerical columns are considered for outlier detection
outlier_columns= list(df.columns[:6])

plt.style.context('dark_background')
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] = 100

df[["acc_y", "label"]].boxplot(by="label", figsize=(20,10))
df[outlier_columns[:3] + ["label"]].boxplot(by="label", figsize=(20,10), layout=(1,3))
df[outlier_columns[3:] + ["label"]].boxplot(by="label", figsize=(20,10), layout=(1,3))
#--------------------------------------------------------------------------------
# These box plots are very good at telling that outliers are there etc. but we cant exactly tell whether a suggested outlier is actually a outlier(cant help in creating outlier col), so we use below function.
# This function takes in df, col and same col but with extra info of values in that col being outlier or not. It creates a time series plot and maps non outliers to blue and outliers tor red.

#IQR method
def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()
    
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

col = "acc_x"
dataset = mark_outliers_iqr(df, col)
plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)

#loop over all colums
for col in outlier_columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)
    
# Thus gyr data has much more outliers than acc data.

#--------------------------------------------------------------------------------

#Chauvenet's criterion(distribution based)
# To use this, it requires our data is normally distributed. 
df[outlier_columns[:3] + ["label"]].plot.hist(by="label", figsize=(20,10), layout=(3,3))
df[outlier_columns[3:] + ["label"]].plot.hist(by="label", figsize=(20,10), layout=(3,3))
# From this, we can see in general we can see its normally distributed except few

def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)
    
# We can see its not hard on data anymore, quite less amt of outliers.

#--------------------------------------------------------------------------------

#Local Outlier Factor(distance based)
 
def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()
  
    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


dataset,outliers,X_scores  = mark_outliers_lof(df, outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(dataset = dataset, col=col, outlier_col =  "outlier_lof", reset_index=True)  
# We see outliers being identified more within the data itself. Because this is distance approach

#--------------------------------------------------------------------------------

#Check outlier grouped by label
label = "dead"
for col in outlier_columns:
    dataset = mark_outliers_iqr(df.query(f"label == '{label}'"), col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)
#this seems to be very strict by graphs

for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df.query(f"label == '{label}'"), col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)
# Seams reasonable now. very less number of outliers so much of data will be conserved

dataset,outliers,X_scores  = mark_outliers_lof(df.query(f"label == '{label}'"), outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(dataset = dataset, col=col, outlier_col =  "outlier_lof", reset_index=True)
    

#--------------------------------------------------------------------------------

#Lets choose the method and deal with outliers
col = "gyr_z"
dataset = mark_outliers_chauvenet(df, col=col)
dataset[dataset["gyr_z_outlier"]]
# We right now set these to NAN
dataset.loc[dataset["gyr_z_outlier"], col] = np.nan

#lets loop
outliers_removed_df = df.copy()
for col in outlier_columns:
    for label in df["label"].unique():
        dataset = mark_outliers_chauvenet(df.query(f"label == '{label}'"), col)
        dataset.loc[dataset[col + "_outlier"], col] = np.nan
        
        outliers_removed_df.loc[(outliers_removed_df["label"] == label), col] = dataset[col]
        
        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"Removed {n_outliers} outliers from {col} for {label}")

outliers_removed_df.info()

#--------------------------------------------------------------------------------

#export the data
outliers_removed_df.to_pickle(r"C:\Users\ARNAV AGARWAL\OneDrive - Indian Institute of Technology Bombay\Desktop\SS\ML\CodingFitnessTracker\data-science-template\data\interim\02_outliers_removed_chauvenets.pkl")

# we still dont know whether chevenets is best or not, lets see.