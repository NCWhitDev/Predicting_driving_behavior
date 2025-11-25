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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Read in csv datasets ---> dataframes
df_test_motion = pd.read_csv("test_motion_data.csv")
df_train_motion = pd.read_csv("train_motion_data.csv")

# combine data sets (train + test)
df_new = pd.concat([df_train_motion, df_test_motion], ignore_index=True)

# split features & target, prepping x and y for set.
X = df_new[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', "Timestamp"]]
y = df_new['Class']


# TO-DO: Cross validation =======================================================================
# Define a "base" model for CV
rf_cv = RandomForestClassifier(n_estimators=200, max_features=2, random_state=7, class_weight='balanced')
# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cv_scores = cross_val_score(rf_cv, X, y, cv=cv, scoring='accuracy')

print("\nCross-Validation Results (5-fold):")
print("Fold accuracies:", cv_scores)
print("Mean accuracy:  {:.4f}".format(cv_scores.mean()))
print("Std accuracy:   {:.4f}".format(cv_scores.std()))

# END ============================================================================================

# test_size tried: 0.2, 0.3, 0.999, 0.001
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)

# rf = Random Forest
# n_estimators = 100 to start since that is SciKit default
# n_estimators: 100, 200, 20, 500
# (Not used) max_depth: 20, 30 - Max amount of times the Random Forest can split down to branches. Size of Tree.
# random_state: controls the randomness by setting the state to be the same ever instances.
# class_weight: Can be used to favor one feature over the other, but we want balanced features.
rf = RandomForestClassifier(n_estimators=200, max_features=2, random_state=7, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print('\nprediction made... analyzing results')
accuracy = accuracy_score(y_test, y_pred) # Compares y prediction to y test to get an accuracy percentage.
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted') # Calculates the precision/recall/fscore of the model.

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy: .4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
# print(f"Loss: {loss:.4f}")

# Shows you results of each type of behavior.
print("\nPer-Class Metrics:")
print(classification_report(y_test, y_pred))

# Visualized Graph using confusion matrix
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
plt.show()