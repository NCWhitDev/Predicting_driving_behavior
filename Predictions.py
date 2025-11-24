# Notes
# Figured a random forest would be a good start, but since it doesn't have memory it can't accurately discern aggressive vs normal driving
# The model predicts slow drivers best, and gets confused with aggressive and normal drivers
# My theory for this is that it is due to the model not having a "memory"
# It is pretty easy to tell if a driver is slow, if they are consistenly accelerating at slow speeds then they are a slow driver
# But it is harder to tell if a driver is aggressive vs normal. Both drivers would go fast, but only an aggressive driver would have sharp changes in acceleration etc.
# However the model doesn't have a memory, therefore it can't tell that a driver often accelerates and decelerates at rapid pace, only that the driver is accelerating/deccelerating
# going to change to a neural network to have a functioning memory. Hopefully that will allow the model to keep better track of the patterns in the data

# nvm neural network was too complicated so we are going back to random forest and aggregating data to see if it works
# i think the default training data was 41% slow drivers, which was most likely imfluencing the model and causing it to misclassify drivers as slow
# concatinating the dataset gives the model more data to randomly chose from

import pandas as pd
#import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Read in csv datasets ---> dataframes
df_test_motion = pd.read_csv("test_motion_data.csv")
df_train_motion = pd.read_csv("train_motion_data.csv")

# combine data since 
df_new = pd.concat([df_test_motion, df_test_motion])

# split features & target
# ommitted timestamp feature because who cares (definitely not the model)
# 11/24 brought back the Timestamp feature and it increased the accuracy to 100% so apparently the model cares
X = df_new[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', "Timestamp"]]
y = df_new['Class']

# X_test = df_test_motion[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']]
# y_test = df_test_motion['Class']

# values used:
# test_size: 0.2, 0.3, 0.999, 0.001
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)

# rf = random forest
# n_estimators = 100 to start since that is SciKit default
# listing values used:
# n_estimators: 100, 200, 20, 500
# max_depth: 20, 30
# random_state:
rf = RandomForestClassifier(n_estimators=200, max_features=2, random_state=7, class_weight='balanced')

# if this prints and then everything stops you know the model froze
print('reached training stage')
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print('prediction made... analyzing results')
accuracy = accuracy_score(y_test, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f"\nModel Performance:")
print(f"Accuracy: {accuracy: .4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
# print(f"Loss: {loss:.4f}")

print("\nPer-Class Metrics:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='GnBu',
            xticklabels=['AGGRESSIVE', 'NORMAL', 'SLOW'],
            yticklabels=['AGGRESIVE', 'NORMAL', 'SLOW'])
plt.ylabel('Actual Behavior Type')
plt.xlabel('Predicted Behavior Type')
plt.title('Consfusion Matrix')
plt.tight_layout()
# plt.savefig('confusion_matrix.png')
plt.show()