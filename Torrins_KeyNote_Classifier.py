# Import all the necessary libraries 
import os # Handling file path and directory operations
import librosa # For necessary audio processing
import numpy as np # For numerical operations
import tensorflow as tf # Developing machine learning 
from tensorflow.keras.models import Sequential # To build a sequential model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # Nueral network layers
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import train_test_split # To split the data into training and test sets

# Dataset Path
DATA_DIR = 'Guitar Dataset'  

# Image Dimensions
TARGET_SHAPE = (128, 128)

# Sample Rate
SR = 44100

# Retrieving class labels
classes = [folder for folder in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, folder))]
classes.sort()  
num_classes = len(classes)

print("Found classes:", classes)

# To create a dictionary to map class labels and numerical indices
class_to_index = {cls: i for i, cls in enumerate(classes)}

# Function to load and process the audio files
def load_and_preprocess_data(data_dir, classes, target_shape=(128, 128), sr=44100):
    data = [] # List to store Spectogram data
    labels = [] # List to store corresponding labels
    
    for cls in classes:
        class_dir = os.path.join(data_dir, cls) # Path to each class folder
        
        for file in os.listdir(class_dir): # To iterate over each audio file
            if file.endswith('.wav'): # Processes only .wav files
                file_path = os.path.join(class_dir, file)
                try:
                    
                    # Load the audio file
                    y, _ = librosa.load(file_path, sr=sr)
                except Exception as e:
                    print("Error loading file:", file_path, e)
                    continue
                
                # converting .wav form to Mel spectogram
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Resize spectrogram to match target shape and add a channel dimension
                mel_spec_db = tf.image.resize(np.expand_dims(mel_spec_db, axis=-1), target_shape)
                
                # Adding the processed data to corresponding labels
                data.append(mel_spec_db.numpy())
                labels.append(class_to_index[cls])
    
    # Converting lists to numpy arrays
    return np.array(data), np.array(labels)


# Loading and processing Data
data, labels = load_and_preprocess_data(DATA_DIR, classes, TARGET_SHAPE, SR)
print("Data shape:", data.shape)     # Print the shape fo the processed data
print("Labels shape:", labels.shape)  # Print the shape of the labeled array 


labels_cat = to_categorical(labels, num_classes=num_classes)

# Splitting the dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_cat, test_size=0.2, random_state=42)

# To the get the input shape of the CNN Network
input_shape = X_train[0].shape

# Building the CNN network
model = Sequential([
    
    # First Convolutional layer with 32 filters and ReLU activation  
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    
    # Second Convolutional layer with 64 filtes and ReLU activation
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(), # Flattening the 2D feature map into a 1D vector
    Dense(128, activation='relu'), # Connected layer 128 nuerons
    Dropout(0.5), # To reduce overfitting
    Dense(num_classes, activation='softmax') # output layer with softmax activation for classification
])

# Compile the model with Adam optimizer and categorical cross-entropy loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Printing model summary 
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

#Evaluating the trained model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", test_acc)

# Saaving the trained model to a file
model.save('note_classifier.h5')
print("Model saved as note_classifier.h5")
