import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from iPython.display import display
#load data
df = pd.read_pickle(r"C:\Users\ARNAV AGARWAL\OneDrive - Indian Institute of Technology Bombay\Desktop\SS\ML\CodingFitnessTracker\data-science-template\data\interim\01_data_processed.pkl")

#lets plot single column
set_df = df[df["set"]==1]
plt.plot(set_df["acc_y"]) # we did the set thing since othwerwise the plot would be too crowded and all sets ka khichdi ban jata
plt.plot(set_df["acc_y"].reset_index(drop=True)) #reset_index(drop=True) is used to reset the index of the dataframe to default index. just exploring. depends on onjective.


#lets plot all exercises
df["label"].unique()
for label in df["label"].unique():
    subset = df[df["label"]==label]
    fig, ax = plt.subplots() #just so that we can customize later
    plt.plot(subset["acc_y"].reset_index(drop=True), label = label)
    plt.legend()
    plt.show()
    
    #lets plot first 100 samples for all
for label in df["label"].unique():
    subset = df[df["label"]==label]
    fig, ax = plt.subplots() #just so that we can customize later
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label = label)
    plt.legend()
    plt.show()
    
    

# Adjust plot settings. TRICK : We will set some settings for mpl and all further plots will follow these settings. Its RC Params

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20,5)
mpl.rcParams["figure.dpi"] = 100


#compare med vs heavy sets of squats 
category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()  #another way of filtering

fig, ax = plt.subplots() #decoration
category_df.groupby(["category"])["acc_y"].plot() #main thing
ax.set_xlabel("Samples") #decoration
ax.set_ylabel("acc_y") #decoration
plt.legend() #decoration
#so we can see from the graph that acc is higher for medium ones(obv its easy to move lighter bar faster)



#compare participants. We want our model to generalize well so we need to see if the data is consistent across participants
participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()  #If we didnt sort, then different participants would be coming in between anywhere. no uniformity
fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_xlabel("Samples")
ax.set_ylabel("acc_y")
plt.legend()
# thus looks kind of same for all participants. So we can say that the data is consistent across participants. E has done a lot of bench press tho :)



#Plot multiple axis(x,y,z for both acc and gyr)
label = "squat"
participant = "A"
all_axis_df = df.query(f"label== '{label}'").query(f"participant == '{participant}'").reset_index() 
fig, ax = plt.subplots()
all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)
ax.set_xlabel("Samples")
ax.set_ylabel("acc_y")
plt.legend()



#Create a loop to plot all combinations per sensor
labels = df["label"].unique()
participants = df["participant"].unique()
#acc
for label in labels:
    for participant in participants:
        all_axis_df = df.query(f"label== '{label}'").query(f"participant == '{participant}'").reset_index() 
        if len(all_axis_df)>0:
            fig,ax = plt.subplots()
            all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)
            ax.set_xlabel("Samples")
            ax.set_ylabel("acc_y")
            plt.title(f"{label} {participant}".title())
            plt.legend()
#gyr
for label in labels:
    for participant in participants:
        all_axis_df = df.query(f"label== '{label}'").query(f"participant == '{participant}'").reset_index() 
        if len(all_axis_df)>0:
            fig,ax = plt.subplots()
            all_axis_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax)
            ax.set_xlabel("Samples")
            ax.set_ylabel("gyr_y")
            plt.title(f"{label} {participant}".title())
            plt.legend()
            



#combine plots in one fig
label = "row"
participant = "A"
combined_plot_df = df.query(f"label== '{label}'").query(f"participant == '{participant}'").reset_index()

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol = 3, fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol = 3, fancybox=True, shadow=True)
ax[1].set_xlabel("samples")




#loop over all combis and export for both sensors
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = df.query(f"label== '{label}'").query(f"participant == '{participant}'").reset_index()
        if len(combined_plot_df)>0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
            combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])
            
            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol = 3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol = 3, fancybox=True, shadow=True)
            ax[1].set_xlabel("samples")
            plt.savefig(rf"C:\Users\ARNAV AGARWAL\OneDrive - Indian Institute of Technology Bombay\Desktop\SS\ML\CodingFitnessTracker\data-science-template\reports\figures\{label.title()} ({participant}).png")
            plt.show()
            
