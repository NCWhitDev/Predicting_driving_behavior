# Core TensorFlow / Keras imports
import tensorflow as tf
import keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# # For data manipulation (if needed)
# import numpy as np
# import pandas as pd

# # For splitting or scaling data (optional)
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler


# Read in csv datasets ---> dataframes
df_test_motion = pd.read_csv("test_motion_data.csv")
df_train_motion = pd.read_csv("train_motion_data.csv")

# split features & target
# ommitted timestamp feature because who cares (definitely not the model)
X_train = df_train_motion[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']]
y_train = df_train_motion['Class'] # Normal/Agrressive

X_test = df_test_motion[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']]
y_test = df_test_motion['Class'] # Normal/Agrressive

# Create RNN model: This guy will solve cancer I swear it!!
RNNModel = Sequential()







# Sources: Recurrent Neural Networks: 
# conda install <PACKAGE_NAME> --name <ENVIRONMENT_NAME>
# Recurrent Neural Network (RNN) model documentation is available for popular deep learning frameworks like TensorFlow (Keras) and PyTorch
    # https://www.d2l.ai/chapter_recurrent-neural-networks
    # https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network
    # https://msalamiitd.medium.com/demystifying-model-architectures-in-tensorflow-a-comprehensive-guide-60393d8fa684
    # https://www.datacamp.com/tutorial/tutorial-for-recurrent-neural-network