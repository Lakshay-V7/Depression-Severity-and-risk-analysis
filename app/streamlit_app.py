import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\saved_model\multimodal_cnn_lstm.keras')

# Load the EEG, Audio data, and labels
eeg_data = np.load(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\eeg_data_aligned.npy')
audio_data = np.load(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\audio_data_cleaned.npy')
labels = np.load(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\depression-prediction\data\processed\labels_cleaned.npy')

# One-hot encode labels
num_classes = len(np.unique(labels))
labels = tf.keras.utils.to_categorical(labels, num_classes)

# Split data for testing (Use proper test data in real scenarios)
X_test_eeg = eeg_data[:80]  # Example: First 80 samples for testing
X_test_audio = audio_data[:80]
y_test = labels[:80]

# Streamlit layout
st.title("Model Performance Visualizations")

# Display Accuracy and Loss Plots
st.subheader("Model Accuracy and Loss")

# If you have a history object saved, load it or use the training logs
# Otherwise, you would need to train again and save history if needed.
history = model.fit(
    [eeg_data, audio_data],
    labels,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    verbose=0  # Suppress output during training for display
)

# Accuracy plot
fig1, ax1 = plt.subplots()
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Loss plot
fig2, ax2 = plt.subplots()
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

# Display accuracy and loss plots
st.pyplot(fig1)
st.pyplot(fig2)

# Confusion Matrix
y_pred = model.predict([X_test_eeg, X_test_audio])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix as heatmap
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes), ax=ax3)
ax3.set_title('Confusion Matrix')
ax3.set_xlabel('Predicted Labels')
ax3.set_ylabel('True Labels')

# Display confusion matrix heatmap
st.subheader("Confusion Matrix")
st.pyplot(fig3)

# Model Evaluation
test_loss, test_accuracy = model.evaluate([X_test_eeg, X_test_audio], y_test, verbose=0)
st.subheader("Model Evaluation")
st.write(f"Test Loss: {test_loss:.4f}")
st.write(f"Test Accuracy: {test_accuracy*100:.2f}%")
