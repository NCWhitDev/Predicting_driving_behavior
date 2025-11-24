from keras import layers, models, Sequential, losses, optimizers
import numpy as np
import pandas as pd

# Read in csv datasets ---> dataframes
df_test_motion = pd.read_csv("test_motion_data.csv")
df_train_motion = pd.read_csv("train_motion_data.csv")