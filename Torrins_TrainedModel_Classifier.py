# Importing the mecessary libraries
import os # For handling file operations and data processing
import numpy as np # For numerical operations
import librosa # For audio processing
import librosa.display # For visualization of the audio files
import tensorflow as tf # For Deep Laerning 
from tensorflow.keras.models import load_model # For loading the trained model
import matplotlib.pyplot as plt # for plotting the result

# Path to the trained model
MODEL_PATH = 'note_classifier.h5'
#Path to the sample music
MUSICAL_PIECE_PATH = 'guitar-purgatory-255374.mp3'
# Sample Rate
SR = 44100
# Target shape for the spetogram image
TARGET_SHAPE = (128, 128)
# Duration for the image processing
SEGMENT_DURATION = 0.5                   

# Loading the trained model 
model = load_model(MODEL_PATH)

# Defining musical notes in a sorted order
classes = sorted(['A2', 'A3', 'A4', 'Asharp2', 'Asharp3', 'Asharp4', 
                  'B2', 'B3', 'B4', 'C3', 'C4', 'C5', 'Csharp3', 'Csharp4', 
                  'Csharp5', 'D2', 'D3', 'D4', 'D5', 'Dsharp2', 'Dsharp3', 
                  'Dsharp4', 'Dsharp5', 'E2', 'E3', 'E4', 'E5', 'F2', 'F3', 
                  'F4', 'F5', 'Fsharp2', 'Fsharp3', 'Fsharp4', 'Fsharp5', 
                  'G2', 'G3', 'G4', 'G5', 'Gsharp2', 'Gsharp3', 'Gsharp4', 'Gsharp5'])
num_classes = len(classes)

# Loading the audio file and re-sampling it to the defined samplig rate
y, sr = librosa.load(MUSICAL_PIECE_PATH, sr=SR)
print(f"Loaded musical piece with {len(y)} samples at {sr} Hz.")

# Detecting note onset
onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# Detecting onset frames
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)

# Converting onset frames into time values 
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

print("Detected onset times (in seconds):", onset_times)

# List to store detected notes and there corresponding time values
predicted_notes = []  

# Iterating thorugh the onset times and processing the corresponding audio samples
for onset_time in onset_times:
    
    # Converting the onset time into sample index
    onset_sample = int(onset_time * sr)
    # Defining sample length in samples
    segment_samples = int(SEGMENT_DURATION * sr)
    
    # Extract the segment of the waveform corresponding to the detected waveform
    y_segment = y[onset_sample:onset_sample + segment_samples]
    # If the extracted segment is shorter than expected, pad with zeros
    if len(y_segment) < segment_samples:
        y_segment = np.pad(y_segment, (0, segment_samples - len(y_segment)), mode='constant')
    
    # Compute the Mel Spectogram of the audio file
    mel_spec = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=128)
    
    # Convert the Mel Spectogram to the decibal scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Resize the spectogram to match the input shape expected by the model
    mel_spec_db = tf.image.resize(np.expand_dims(mel_spec_db, axis=-1), TARGET_SHAPE)
    
    # Expand the dimensions to match the input format expected by the model
    input_data = np.expand_dims(mel_spec_db.numpy(), axis=0)
    
    # Predicting the musical note using trained model
    predictions = model.predict(input_data)
    # getting the index of the predicted note
    predicted_index = np.argmax(predictions)
    #Retrieve the corresponding note label
    predicted_note = classes[predicted_index]
    # Store the onset time and predicted note
    predicted_notes.append((onset_time, predicted_note))

#display the detected notes with corresponding onset times 
print("\nDetected notes in the musical piece:")
for time_val, note in predicted_notes:
    print(f"Time: {time_val:.2f}s -> Note: {note}")

# ploting the onsets on the waveform

# Dimensions of the plot 
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.vlines(onset_times, -1, 1, color='r', linestyle='--', label='Detected Onsets')
plt.title("Waveform with Detected Onsets")
plt.legend()
plt.show()
