import gradio as gr
import librosa
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def preprocess_audio(file_path):
    """Loads and processes audio file into MFCC features."""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        if audio.size == 0:
            print(f"Warning: {file_path} is empty and will be skipped.")
            return None
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)  # Get mean of MFCCs across time
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Define directories for real and fake audio samples
real_dir = '/Users/ms/Downloads/Dataset_Speech_Assignment/Real'
fake_dir = '/Users/ms/Downloads/Dataset_Speech_Assignment/Fake'
real_features, fake_features = [], []

# Process audio files to extract features
for directory, feature_list in [(real_dir, real_features), (fake_dir, fake_features)]:
    for filename in os.listdir(directory):
        if filename.endswith(('.mp3', '.wav')):
            file_path = os.path.join(directory, filename)
            features = preprocess_audio(file_path)
            if features is not None:
                feature_list.append(features)

# Combine and prepare the data
X = np.array(real_features + fake_features)
y = np.array([1] * len(real_features) + [0] * len(fake_features))  # Labels: 1 for real, 0 for fake

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the trained model
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Define a Gradio interface to predict using the uploaded file
def predict(file_path):
    features = preprocess_audio(file_path)
    if features is not None:
        features = np.array([features])
        prediction = model.predict(features)
        return prediction[0]
    else:
        return "Error processing file"

# Setup Gradio interface
iface = gr.Interface(fn=predict, inputs="file", outputs="text")
iface.launch(share=True)

