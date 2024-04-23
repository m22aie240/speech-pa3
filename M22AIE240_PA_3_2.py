# Working with the 2-second dataset as mentioned in part 4 of the assignment



import librosa
import numpy as np
import os

def load_data(directory):
    X, y = [], []
    for label in ['real', 'fake']:
        class_dir = os.path.join(directory, label)
        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            if file_path.endswith('.wav'):
                audio, sr = librosa.load(file_path, sr=None)
                # Ensure you're using keyword arguments for mfcc
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                mfccs = np.mean(mfccs.T, axis=0)  # Averaging over time
                X.append(mfccs)
                y.append(1 if label == 'real' else 0)
    return np.array(X), np.array(y)



train_dir = '/Users/ms/Downloads/for-2seconds/training'
test_dir = '/Users/ms/Downloads/for-2seconds/testing'
val_dir = '/Users/ms/Downloads/for-2seconds/validation'

X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)
X_val, y_val = load_data(val_dir)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)


from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Predict and calculate AUC
y_pred_probs_test = model.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, y_pred_probs_test)

# Calculate EER
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs_test)
eer_test = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

print(f'AUC on Test Set: {auc_test}, EER on Test Set: {eer_test}')



import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


