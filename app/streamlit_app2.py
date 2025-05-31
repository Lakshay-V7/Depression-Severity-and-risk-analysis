import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from io import BytesIO
import librosa

# Load the pre-trained model
model = tf.keras.models.load_model(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\saved_model\multimodal_cnn_lstm.keras')

# Streamlit Layout
st.title("Depression Prediction Model - Live Results")

# File upload sections for EEG and Audio data
st.header("Upload EEG Data (Numpy format .npy)")
eeg_file = st.file_uploader("Choose EEG File", type=["npy"], key="eeg_file")

st.header("Upload Audio Data (Numpy format .npy)")
audio_file = st.file_uploader("Choose Audio File", type=["npy"], key="audio_file")

# Function to load .npy files from uploaded data
def load_data(file):
    if file is not None:
        data = np.load(file)
        return data
    else:
        return None

# Initialize the uploaded EEG and Audio Data
eeg_data = None
audio_data = None

if eeg_file is not None:
    eeg_data = load_data(eeg_file)
    st.write("EEG Data shape: ", eeg_data.shape)

if audio_file is not None:
    audio_data = load_data(audio_file)
    st.write("Audio Data shape: ", audio_data.shape)

# Button to make predictions after data upload
if st.button('Make Prediction'):
    if eeg_data is not None and audio_data is not None:
        # Make predictions using the loaded model
        predictions = model.predict([eeg_data, audio_data])
        predicted_class = np.argmax(predictions, axis=1)

        # Show the predicted class
        st.subheader("Prediction Result")
        st.write(f"Predicted Class (Depression Severity): {predicted_class[0]}")

        # Display the confidence scores
        st.write("Confidence Scores: ")
        st.write(predictions)

        # Plot the confusion matrix for visualization (using the sample test data)
        y_test = np.load(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\labels_cleaned.npy')
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)  # Assuming 2 classes (depressed and non-depressed)
        
        # Example test data (Use actual test data in a real scenario)
        y_pred = model.predict([eeg_data, audio_data])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_classes)

        # Plot confusion matrix as heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(2), yticklabels=np.arange(2), ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')

        # Display confusion matrix heatmap
        st.subheader("Confusion Matrix")
        st.pyplot(fig)
    else:
        st.write("Please upload both EEG and Audio data to make a prediction.")

# Model Evaluation (using existing test data)
if st.button("Evaluate Model"):
    # Evaluation on existing test data
    eeg_test = np.load(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\eeg_data_aligned.npy')[:80]
    audio_test = np.load(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\audio_data_cleaned.npy')[:80]
    labels_test = np.load(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\labels_cleaned.npy')[:80]

    # One-hot encode the test labels
    labels_test = tf.keras.utils.to_categorical(labels_test, num_classes=2)

    # Model evaluation
    test_loss, test_accuracy = model.evaluate([eeg_test, audio_test], labels_test, verbose=0)
    st.subheader("Model Evaluation")
    st.write(f"Test Loss: {test_loss:.4f}")
    st.write(f"Test Accuracy: {test_accuracy*100:.2f}%")
