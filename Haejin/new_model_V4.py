"""
SNAP ERROR RATE MODEL
"""
# 1. converted to this python code
# 2. commented out the columns in feature engineer section; row 43-74 and included them into `cols_to_select`
# 3. added class weights for imbalanced classes in section #6

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ========== SECTION 1: MODEL TRAINING ==========

print("="*70)
print("SNAP ERROR RATE MODEL - TRAINING")
print("="*70)

# ── 1. LOAD DATA ────────────────────────────────────

mydata = pd.read_csv("~/github-repos/SNAP/Data_UVA_SNAProject.csv")
print(f"\nData: {mydata.shape[0]:,} cases, {mydata.shape[1]} columns")

# ── 2. SELECT & ENGINEER FEATURES ───────────────────────────

cols_to_select = [
    'error_target', 'case_ern_incm_amt', 'case_uern_incm_amt', 'case_slf_ern_incm_amt',
    'unit_disable', 'unit_children', 'is_married', 'is_old', 'case_nonenglish',
    'group_region', 'med_age', 'family_size', 'grouprace', 'max_educ',
    'dss_tenure', 'tot_shl_cost_amt', 'homeless_allow_amt',
    'med_exp_amt', 'bas_utl_allow_amt', 'chld_sup_exp_amt',
    'aboveavg_uern_fips', 'aboveavg_ern_fips', 'aboveavg_shl_fips',
    'move_flag', 'Issuance', 'Error', 'caseload',
    'total_income', 'round_income', 'round_income_un', 'share_rent',
    'flag_inchh', 'inc_cap', 'max_snap', 'share_issuance'
]

mydata = mydata[cols_to_select].copy()

# Feature engineering
# These below 10 columns are already in the source table.
# mydata['total_income'] = (mydata['case_ern_incm_amt'] + 
#                          mydata['case_uern_incm_amt'] + 
#                          mydata['case_slf_ern_incm_amt'])

#mydata['round_income'] = (
#    (mydata['case_ern_incm_amt'] != 0) & 
#    (mydata['case_ern_incm_amt'] == np.round(mydata['case_ern_incm_amt']))
#).astype(int)

#mydata['round_income_un'] = (
#    (mydata['case_uern_incm_amt'] != 0) & 
#    (mydata['case_uern_incm_amt'] == np.round(mydata['case_uern_incm_amt']))
#).astype(int)

#mydata['share_rent'] = mydata['tot_shl_cost_amt'] / (mydata['total_income'] + 1)

#mydata['flag_inchh'] = (
#    (mydata['family_size'] >= 3) & 
#    ((mydata['case_ern_incm_amt'] + mydata['case_uern_incm_amt']) >= 1500)
#).astype(int)

#mydata['inc_cap'] = mydata['total_income'] / mydata['family_size']
mydata['caseload_cap'] = mydata['caseload'] / (mydata['dss_tenure'] + 1)

#snap_benefits = {
#    1: 298, 2: 546, 3: 785, 4: 994, 5: 1183,
#    6: 1421, 7: 1571, 8: 1789
#}
#mydata['max_snap'] = mydata['family_size'].map(snap_benefits)
#mydata['share_issuance'] = mydata['Issuance'] / mydata['max_snap']

print("most feature engineering already done and provided in source data")

# ── 3. ENCODE CATEGORICALS ──────────────────────────────────

#factor_vars <- c(
#  "group_region", "grouprace", "is_married", "is_old",
#  "case_nonenglish", "max_educ", "move_flag",
#  "aboveavg_ern_fips", "aboveavg_uern_fips", "aboveavg_shl_fips"
#)
#mydata[factor_vars] <- lapply(mydata[factor_vars], as.factor)

# One-hot encode multi-level factors; binary factors stay as 0/1 numeric
region_ohe = pd.get_dummies(mydata['group_region'], prefix='group_region')
race_ohe = pd.get_dummies(mydata['grouprace'], prefix='grouprace')
educ_ohe = pd.get_dummies(mydata['max_educ'], prefix='max_educ')

mydata = mydata.drop(columns=['group_region', 'grouprace', 'max_educ', 'max_snap'])
mydata = pd.concat([mydata, region_ohe, race_ohe, educ_ohe], axis=1)

print(f"One-hot encoding complete: {mydata.shape[1]} total features")
print(mydata.head())

# ── 4. SPLIT TARGET / FEATURES ──────────────────────────────

np.random.seed(1234)
mydata = mydata.sample(frac=1, random_state=1234).reset_index(drop=True) # shuffle (reordering)
# no under sampling or over sampling in this script?

target = mydata['error_target'].values
error_dollars = mydata['Error'].values # preserve dollar errors for later

feature_matrix = mydata.drop(columns=['error_target', 'Error']).values
feature_cols = mydata.drop(columns=['error_target', 'Error']).columns.tolist()
print(feature_matrix)
print(feature_cols)

# ── 5. TRAIN / TEST SPLIT (70 / 30) ─────────────────────────

n_samples = len(feature_matrix)
print(f"Total samples: {n_samples:,}")

n_train = int(np.round(n_samples * 0.7)) # 70/30 split

train_data = feature_matrix[:n_train]
train_labels = target[:n_train]

test_data = feature_matrix[n_train:]
test_labels = target[n_train:]
test_errors = error_dollars[n_train:] # dollar errors aligned to test set

print(f"\n✓ Train/Test split (70/30):")
print(f"  - Training set: {len(train_labels):,} cases")
print(f"  - Test set: {len(test_labels):,} cases")

# ── 6. TRAIN XGBOOST WITH CLASS WEIGHTS ──────────────────────────────────

# Calculate class weights to handle imbalance
n_negative = (train_labels == 0).sum()
n_positive = (train_labels == 1).sum()
scale_pos_weight = n_negative / n_positive

print(f"\nClass imbalance handling:")
print(f"  - Negative (0): {n_negative:,}")
print(f"  - Positive (1): {n_positive:,}")
print(f"  - Scale pos weight: {scale_pos_weight:.2f}")

dtrain = xgb.DMatrix(train_data, label=train_labels)
dtest = xgb.DMatrix(test_data, label=test_labels)

params = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': scale_pos_weight,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

watchlist = [(dtrain, 'train'), (dtest, 'test')]

# mirror the original R script; lower complexity, smaller learning rate, L1 regularization
train_params = params.copy()
train_params.update({'max_depth': 4, 'eta': 0.05, 'alpha': 1})

model = xgb.train(
    train_params,
    dtrain,
    num_boost_round=200,
    evals=watchlist,
    early_stopping_rounds=10,
    verbose_eval=False
)

print(f"✓ XGBoost model trained ({model.best_iteration} rounds)")

# ── 6. EVALUATE ─────────────────────────────────────────────

pred = model.predict(dtest)
pred_class = (pred > 0.5).astype(int)

cm = confusion_matrix(test_labels, pred_class)
fpr, tpr, _ = roc_curve(test_labels, pred)
auc_value = auc(fpr, tpr)

# Calculate metrics
tn, fp, fn, tp_count = cm.ravel()
sensitivity = tp_count / (tp_count + fn) if (tp_count + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp_count / (tp_count + fp) if (tp_count + fp) > 0 else 0

print(f"\n{'='*70}")
print("MODEL PERFORMANCE METRICS")
print(f"{'='*70}")
print(f"Sensitivity (Catch errors):   {sensitivity:.2%}  (catches {sensitivity:.0%} of actual errors)")
print(f"Specificity (Clear valid):    {specificity:.2%}  (correctly clears {specificity:.0%})")
print(f"Precision (True errors/flag): {precision:.2%}  ({precision:.0%} of flagged are true errors)")
print(f"AUC-ROC:                      {auc_value:.4f}")

print(f"\nConfusion Matrix:")
print(f"{'':25s} Predicted=0  Predicted=1")
print(f"Actual=0 (no error)      {tn:6d}       {fp:6d}")
print(f"Actual=1 (error)         {fn:6d}       {tp_count:6d}")

# ── 7. ERROR DOLLAR ANALYSIS ────────────────────────────────

results = pd.DataFrame({
    'actual_error_flag': test_labels,
    'pred_flag': pred_class,
    'pred_prob': pred,
    'error_dollars': test_errors
})

def classify_outcome(row):
    if row['actual_error_flag'] == 1 and row['pred_flag'] == 1:
        return "True Positive"
    elif row['actual_error_flag'] == 0 and row['pred_flag'] == 1:
        return "False Positive"
    elif row['actual_error_flag'] == 1 and row['pred_flag'] == 0:
        return "False Negative"
    else:
        return "True Negative"

results['outcome'] = results.apply(classify_outcome, axis=1)

outcome_summary = results.groupby('outcome').agg({
    'error_dollars': ['count', 'mean', 'sum']
}).round(2)
outcome_summary.columns = ['n_cases', 'mean_error', 'total_error']
outcome_summary = outcome_summary.reset_index()

print(f"\n{'='*70}")
print("ERROR DOLLAR ANALYSIS (Test Set)")
print(f"{'='*70}")

for _, row in outcome_summary.iterrows():
    outcome = row['outcome']
    n_cases = int(row['n_cases'])
    mean_err = row['mean_error']
    total_err = row['total_error']
    
    if outcome == "True Positive":
        print(f"\n✓ TRUE POSITIVES (Correctly caught errors):")
        print(f"    Cases:              {n_cases:,}")
        print(f"    Avg error per case: ${mean_err:,.2f}")
        print(f"    Total error caught: ${total_err:,.0f}")
    elif outcome == "False Positive":
        print(f"\n⚠ FALSE POSITIVES (False alarms):")
        print(f"    Cases:              {n_cases:,}")
        print(f"    Avg error per case: ${mean_err:,.2f}")
    elif outcome == "False Negative":
        print(f"\n✗ FALSE NEGATIVES (Missed errors):")
        print(f"    Cases:              {n_cases:,}")
        print(f"    Avg error per case: ${mean_err:,.2f}")
        print(f"    Total error missed: ${total_err:,.0f}")
    else:
        print(f"\n✓ TRUE NEGATIVES (Correctly cleared):")
        print(f"    Cases:              {n_cases:,}")

# Summary stats
total_error = np.sum(np.abs(results['error_dollars']))
flagged_error = np.sum(np.abs(results[results['pred_flag'] == 1]['error_dollars']))
missed_error = np.sum(np.abs(results[results['pred_flag'] == 0]['error_dollars']))
n_flagged = np.sum(pred_class)

print(f"\n{'='*70}")
print("OPERATIONAL IMPACT")
print(f"{'='*70}")
print(f"\nTotal test set cases:               {len(test_labels):,}")
print(f"Total error in test set:            ${total_error:,.0f}")
print(f"\nCases flagged for review (>50%):   {n_flagged:,} ({100*n_flagged/len(test_labels):.1f}%)")
print(f"Error captured by flagging:         ${flagged_error:,.0f} ({100*flagged_error/total_error:.1f}%)")
print(f"Error missed (not flagged):         ${missed_error:,.0f} ({100*missed_error/total_error:.1f}%)")

if n_flagged > 0:
    print(f"Expected error per case reviewed:   ${flagged_error/n_flagged:,.2f}")

# Threshold analysis
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
print(f"\n{'='*70}")
print("THRESHOLD ANALYSIS")
print(f"{'='*70}")
print(f"\nThreshold  Cases Flagged  % of Test Set   Est. Error Captured")
print(f"-" * 65)
for t in thresholds:
    n = np.sum(pred >= t)
    pct = 100 * n / len(pred)
    captured = np.sum(np.abs(results[results['pred_prob'] >= t]['error_dollars']))
    pct_captured = 100 * captured / total_error if total_error > 0 else 0
    print(f"  {t:.0%}        {n:6,}          {pct:6.1f}%          {pct_captured:6.1f}%")

print(f"\n{'='*70}\n")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {auc_value:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SNAP Error Detection Model')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('/Users/haejin/github-repos/SNAP/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ ROC curve saved to: /Users/haejin/github-repos/SNAP/roc_curve.png")

# Probability histogram
plt.figure(figsize=(10, 6))
plt.hist(pred[pred > 0.5], bins=30, color='steelblue', edgecolor='white')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Predicted Error Probability Distribution (> 50%)')
plt.axvline(0.60, linestyle='--', color='darkred', alpha=0.7, label='60%')
plt.axvline(0.70, linestyle='--', color='darkred', alpha=0.7, label='70%')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.savefig('/Users/haejin/github-repos/SNAP/probability_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Histogram saved to: /Users/haejin/github-repos/SNAP/probability_histogram.png")

print("\n✓ Training complete!")