# Notes
# Figured a random forest would be a good start, but since it doesn't have memory it can't accurately discern aggressive vs normal driving
# The model predicts slow drivers best, and gets confused with aggressive and normal drivers
# My theory for this is that it is due to the model not having a "memory"
# It is pretty easy to tell if a driver is slow, if they are consistenly accelerating at slow speeds then they are a slow driver
# But it is harder to tell if a driver is aggressive vs normal. Both drivers would go fast, but only an aggressive driver would have sharp changes in acceleration etc.
# However the model doesn't have a memory, therefore it can't tell that a driver often accelerates and decelerates at rapid pace, only that the driver is accelerating/deccelerating
# going to change to a neural network to have a functioning memory. Hopefully that will allow the model to keep better track of the patterns in the data

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Read in csv datasets ---> dataframes
df_test_motion = pd.read_csv("test_motion_data.csv")
df_train_motion = pd.read_csv("train_motion_data.csv")

# split features & target
# ommitted timestamp feature because who cares (definitely not the model)
X_train = df_train_motion[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']]
y_train = df_train_motion['Class']

X_test = df_test_motion[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']]
y_test = df_test_motion['Class']

# rf = random forest
# n_estimators = 100 to start since that is SciKit default
# listing values used:
# n_estimators: 100, 200, 20
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['AGGRESSIVE', 'NORMAL', 'SLOW'],
            yticklabels=['AGGRESIVE', 'NORMAL', 'SLOW'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Consfusion Matrix')
plt.tight_layout()
#plt.savefig('confusion_matrix.png')
plt.show()