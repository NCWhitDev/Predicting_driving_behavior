# Predicting dangerous driving behavior
Aggressive driving is strongly associated with fatal crashes, accounting for >50% of deadly crashes
in one analysis of a recent 4-year period. This makes early detection useful for safety intervention
and for future autonomous driver coaching.

Used Dataset: https://www.kaggle.com/datasets/outofskills/driving-behavior?resource=download

Metric || Meaning || Why its useful?
=====================================
Accuracy || x amount of predictions || Provides an overall sense of model performance.
<br>
Precision || out of x samples predicted as a given class || High precision means fewer false claims.
<br>
Recall || How many true samples did the model find || High recall means fewer missed aggressive.
<br>
F1-Score || Harmonic mean of precision and recall || balance false pos and false neg.
<br>
Loss || How confident the model is on predictions || Helps track training progress and detects flaws in model.
