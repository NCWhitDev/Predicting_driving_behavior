import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df_train_motion = pd.read_csv("train_motion_data.csv")
df_test_motion  = pd.read_csv("test_motion_data.csv")

for df in (df_train_motion, df_test_motion):
    # Acceleration magnitude
    df["acc_mag"] = np.sqrt(df["AccX"]**2 + df["AccY"]**2 + df["AccZ"]**2)
    # Gyro magnitude
    df["gyro_mag"] = np.sqrt(df["GyroX"]**2 + df["GyroY"]**2 + df["GyroZ"]**2)

feature_cols = [
    "AccX", "AccY", "AccZ",
    "GyroX", "GyroY", "GyroZ",
    "Timestamp",
    "acc_mag", "gyro_mag"
]

X_train_full = df_train_motion[feature_cols]
y_train_full = df_train_motion["Class"]

X_test_final = df_test_motion[feature_cols]
y_test_final = df_test_motion["Class"]


# CROSS-VALIDATION ON TRAINING SET ONLY
rf_cv = RandomForestClassifier(
    n_estimators=200,
    max_features=2,
    random_state=7,
    class_weight='balanced'
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

cv_scores = cross_val_score(
    rf_cv,
    X_train_full,
    y_train_full,
    cv=cv,
    scoring='accuracy'
)

print("\nCross-Validation Results on TRAIN set (5-fold):")
print("Fold accuracies:", cv_scores)
print("Mean accuracy:  {:.4f}".format(cv_scores.mean()))
print("Std accuracy:   {:.4f}".format(cv_scores.std()))


# TRAIN FINAL MODEL ON ALL TRAIN DATA
rf = RandomForestClassifier(
    n_estimators=200,
    max_features=2,
    random_state=7,
    class_weight='balanced'
)

rf.fit(X_train_full, y_train_full)


# EVALUATE ON TRUE TEST SET
y_pred = rf.predict(X_test_final)

print('\nPrediction made... analyzing results')
accuracy = accuracy_score(y_test_final, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(
    y_test_final, y_pred, average='weighted'
)

print(f"\nModel Performance on TEST set:")
print(f"Accuracy:  {accuracy: .4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nPer-Class Metrics:")
print(classification_report(y_test_final, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_final, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='GnBu',
    xticklabels=['AGGRESSIVE', 'NORMAL', 'SLOW'],
    yticklabels=['AGGRESSIVE', 'NORMAL', 'SLOW']
)
plt.ylabel('Actual Behavior Type')
plt.xlabel('Predicted Behavior Type')
plt.title('Confusion Matrix - Simple RF Model')
plt.tight_layout()
plt.show()

#  model is literally predicting NORMAL for every single example....