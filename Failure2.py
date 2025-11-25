import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Read in csv datasets ---> dataframes
df_test_motion = pd.read_csv("test_motion_data.csv")
df_train_motion = pd.read_csv("train_motion_data.csv")

# FEATURES AND TARGET
feature_cols = ['AccX', 'AccY', 'AccZ',
                'GyroX', 'GyroY', 'GyroZ', 'Timestamp']

# For cross-validation we use ONLY the training data
X_cv = df_train_motion[feature_cols]
y_cv = df_train_motion['Class']

# Cross validation block
rf_cv = RandomForestClassifier(
    n_estimators=200,
    max_features=2,
    random_state=7,
    class_weight='balanced'
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

cv_scores = cross_val_score(
    rf_cv,
    X_cv,
    y_cv,
    cv=cv,
    scoring='accuracy'
)

print("\nCross-Validation Results (5-fold on TRAIN set):")
print("Fold accuracies:", cv_scores)
print("Mean accuracy:  {:.4f}".format(cv_scores.mean()))
print("Std accuracy:   {:.4f}".format(cv_scores.std()))


# Final train + test
# Train on ALL training data
X_train = df_train_motion[feature_cols]
y_train = df_train_motion['Class']
# Test on the held-out test file
X_test = df_test_motion[feature_cols]
y_test = df_test_motion['Class']
rf = RandomForestClassifier(
    n_estimators=200,
    max_features=2,
    random_state=7,
    class_weight='balanced'
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print('\nprediction made... analyzing results')
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average='weighted'
)

print(f"\nModel Performance on TEST set:")
print(f"Accuracy:  {accuracy: .4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nPer-Class Metrics:")
print(classification_report(y_test, y_pred))

# Visualized Graph using confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
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
plt.title('Confusion Matrix - Simple RF Model (Fixed)')
plt.tight_layout()
plt.show()

# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!
# WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!! WHY NORMAL!!!