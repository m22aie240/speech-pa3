import wandb

# Initialize a new run
wandb.init(project="audio_speaker_verification_PA3", entity="m22aie240")

# Optionally, you can add configuration details
config = wandb.config
config.learning_rate = 0.01  # Example configuration

from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Predict probabilities on the test dataset
probabilities = model.predict_proba(X_test)[:, 1]  # Modify according to your model's output

# Calculate AUC and EER for both the datasets
auc_score_cd = roc_auc_score(y_test, probabilities)
auc_score_for = roc_auc_score(y_test, y_pred_probs_test)
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
eer_cd = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred_probs_test)
eer_for = brentq(lambda x: 1. - x - interp1d(fpr1, tpr1)(x), 0., 1.)

# Log metrics
wandb.log({"AUC_CD": auc_score_cd, "EER_CD": eer_cd, "AUC_FOR": auc_score_for, "EER_FOR": eer_for })

# Finish the run
wandb.finish()

