SNAP Payment Error Detection Model
===================================

Author: Humaid Billoo
Date:   June 17, 2026
Data:   Virginia DSS QC sample -- October 2021 through May 2025 (42 months)


WHAT THIS DOES
--------------
Virginia DSS processes thousands of SNAP (food stamp) benefit cases each month.
A subset contain payment errors -- overissuances, underissuances, or eligibility
mistakes -- which drive up the federal Payment Error Rate (PER). States with high
PER face financial penalties.

This model scores each case with a predicted probability of containing an error,
so reviewers can focus limited audit capacity on the highest-risk cases first
rather than reviewing randomly.


DATASET
-------
Source file       : Data_UVA_SNAProject.csv
Total rows        : 3,510 (after filtering)
Removed           : 148 cases marked "2. Not subject to review" -- exempt from error by policy
Date range        : 2021m10 to 2025m5 (42 months)
Error cases       : 1,357 (38.7%)
No-error cases    : 2,153 (61.3%)
Incomplete reviews: 802 -- labels may be noisy
Feature matrix    : 3,510 rows x 51 features (64 columns after one-hot encoding)
Categorical feats : va_region, grouprace, job_category, max_educ


MODEL
-----
Algorithm         : XGBoost binary classifier
Imbalance handling: SMOTENC -- applied inside each CV fold on training data only
Validation        : 5-fold stratified cross-validation
Primary threshold : 0.65

Hyperparameters:
  n_estimators=500, max_depth=4, learning_rate=0.03,
  min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
  reg_alpha=2.0, reg_lambda=2.0, eval_metric='aucpr'


CROSS-VALIDATION RESULTS
------------------------
Fold   ROC-AUC   PR-AUC   Prec@0.65   Recall@0.65   F1@0.65   Flagged
  1    0.7557    0.6054    0.6171      0.3971        0.4832    175
  2    0.7470    0.5893    0.6272      0.3897        0.4807    169
  3    0.7475    0.6009    0.6467      0.3579        0.4608    150
  4    0.7445    0.5968    0.5987      0.3469        0.4393    157
  5    0.7598    0.6020    0.6761      0.3542        0.4649    142
Mean   0.7509    0.5989    0.6332      0.3692        0.4658    158.6
Std    0.0065    0.0062    0.0296      0.0226        0.0177     13.5

AUC std = 0.0065 -- model is stable across folds.


PERFORMANCE AT PRIMARY THRESHOLD (0.65)
----------------------------------------
ROC-AUC       : 0.7504  (model ranks a random error above a random clean case 75% of the time)
PR-AUC        : 0.5950  (vs. 0.3866 baseline -- 1.54x lift over random)
Precision     : 0.6318  (6.3 of every 10 flagged cases are real errors)
Recall        : 0.3692  (catches ~37% of all errors in the caseload)
Specificity   : 0.8644  (86% of clean cases are correctly left unflagged)
F1            : 0.4660
MCC           : 0.2720  (meaningful above random; 0 = random, 1 = perfect)
Cases flagged : 793     (22.6% of caseload)
Brier Score   : 0.1952
Log Loss      : 0.5743

Confusion matrix at threshold = 0.65:
                   Predicted No Error   Predicted Error
Actual No Error    TN = 1,861           FP = 292  (wasted review time)
Actual Error       FN = 856  (missed)   TP = 501  (correctly flagged)


THRESHOLD OPTIONS
-----------------
Threshold   Precision   Recall   F1      Flagged    % Caseload
  0.50      0.600       0.668    0.632   1,510      43.0%
  0.65 *    0.632       0.369    0.466     793      22.6%  <- PRIMARY
  0.70      0.620       0.232    0.338     508      14.5%
  0.75      0.642       0.130    0.216     274       7.8%

Use 0.65 for standard operation. Use 0.70 if review capacity is severely limited.


CAPACITY PLANNING (TOP-N% REVIEW)
-----------------------------------
If the team can only review a fixed number of cases, work from the top of the
score list down:

% Reviewed   N Cases   Threshold   Precision   Errors Found   % of All Errors   Lift
  5%           176       0.771       0.614        108            8.0%            1.59x
 10%           351       0.733       0.627        220           16.2%            1.62x
 20%           702       0.665       0.632        444           32.7%            1.64x
 30%         1,053       0.603       0.629        662           48.8%            1.63x


TOP FEATURES
------------
Feature              Importance   Meaning
total_income         0.0817       Sum of earned + unearned + self-employment income
Issuance             0.0674       SNAP benefit dollar amount actually issued
is_separated         0.0561       Household separation status
case_nonenglish      0.0434       Non-English speaking household flag
is_old               0.0404       Elderly household member present
inc_cap              0.0398       Income per household member
is_disable           0.0317       Disabled household member present
shelter_flag         0.0289       Household claims a shelter cost deduction
complex_hh           0.0242       Count of complexity flags: elderly + disabled + children (0-3)
share_issuance       0.0232       Benefit issued as % of household maximum



