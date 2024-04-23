#!/usr/bin/env python
# coding: utf-8



from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from sklearn.model_selection import train_test_split

# Initialize the model with a pre-defined configuration
config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC(config)



#Mohit Chandra Saxena Speech PA-3



import torch
import torch.nn as nn

# Define the model architecture here
class YourModelClass(nn.Module):
    def __init__(self):
        super(YourModelClass, self).__init__()
        # Your model layers go here

    def forward(self, x):
        # Define the forward pass
        return x

# Create an instance of your model
model = YourModelClass()

# Path to our pretrained model's state dict
state_dict_path = '/Users/ms/Downloads/LA_model.pth'

# Load the state dict onto the CPU
state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))

# Load the state dict into the model
model.load_state_dict(state_dict, strict=False)


# Set the model to evaluation mode
model.eval()


import librosa
import numpy as np
import os

# Define a function to load an audio file and preprocess it
def load_and_preprocess_audio(file_path):
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None)
        if audio.size == 0:  # Check if the audio is empty
            print(f"Warning: {file_path} is empty and will be skipped.")
            return None
        # Compute MFCC features from the audio
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        # Transpose the result to have a shape of (n_samples, n_features)
        mfccs = mfccs.T
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

real_dir = '/Users/ms/Downloads/Dataset_Speech_Assignment/Real'
fake_dir = '/Users/ms/Downloads/Dataset_Speech_Assignment/Fake'
real_features = []
fake_features = []

for filename in os.listdir(real_dir):
    if filename.endswith('.mp3') or filename.endswith('.wav'):
        file_path = os.path.join(real_dir, filename)
        mfccs = load_and_preprocess_audio(file_path)
        if mfccs is not None:
            real_features.append(mfccs)
        else:
            print(f"Skipped None from real: {filename}")  # Debug message

for filename in os.listdir(fake_dir):
    if filename.endswith('.mp3') or filename.endswith('.wav'):
        file_path = os.path.join(fake_dir, filename)
        mfccs = load_and_preprocess_audio(file_path)
        if mfccs is not None:
            fake_features.append(mfccs)
        else:
            print(f"Skipped None from fake: {filename}")  # Debug message

X = real_features + fake_features
print(f"Total features collected: {len(X)}")  # Debug message
X = [x for x in X if x is not None]  # Double-check filtering
print(f"Total features after filtering: {len(X)}")  # Debug message



from sklearn.metrics import roc_curve, auc

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    return eer

def calculate_auc(y_true, y_scores):
    return auc(fpr, tpr)



import numpy as np

max_length = max(len(mfcc) for mfcc in X)

# As we have filtered out None values as shown earlier
if X:  # Check if X is not empty
    max_length = max(len(mfcc) for mfcc in X)

    def pad_mfcc(mfcc, max_length):
        padding = max_length - len(mfcc)
        if padding > 0:
            return np.pad(mfcc, ((0, padding), (0, 0)), mode='constant', constant_values=(0, 0))
        return mfcc

    X_padded = np.array([pad_mfcc(mfcc, max_length) for mfcc in X])
else:
    print("No valid data to process")


#X_padded = np.array([pad_mfcc(mfcc, max_length) for mfcc in X])


# As `real_features` and `fake_features` are lists of MFCCs from the preprocessing step
X = real_features + fake_features

# Find the maximum length of the MFCC arrays
max_length = max(len(mfcc) for mfcc in X)

# Pad each MFCC to have the same length
X_padded = np.array([pad_mfcc(mfcc, max_length) for mfcc in X])

y = [1] * len(real_features) + [0] * len(fake_features)  # 1 for Real, 0 for Fake

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Convert lists to numpy arrays for compatibility with scikit-learn and PyTorch
#X = np.array(X)
#y = np.array(y)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# Assuming X_padded is your data and it has an extra dimension from padding
# Flatten the data
X_flattened = np.array([x.flatten() for x in X_padded])

# Now split the flattened data
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42)


# Initialize the model
model = LogisticRegression(max_iter=1000)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_probs = model.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class



from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

# Calculate the AUC
auc = roc_auc_score(y_test, y_pred_probs)
print(f'AUC: {auc}')

# Calculate the EER
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
print(f'EER: {eer}')



#Plot the AUC Curve 
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

