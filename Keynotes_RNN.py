# Importing all the necesary files
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Defining all the constants
IMAGE_SIZE = (128, 128) # Size of the spectogram images
BATCH_SIZE = 32 # Batch size for training
EPOCHS = 30 # Number of training epochs
DATASET_PATH = "specto1" # path to spectogram images
MODEL_SAVE_PATH = "guitar_note_rnn_classifier.h5" # path to save the trained model

# function to convert .wav format to a spectogram of .png format
def audio_to_spectrogram(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory) # creating an output directory if there is none

    for root, _, files in os.walk(input_directory):
        for file_name in files:
            if file_name.endswith('.wav'): # processing only .wav file
                try:
                    file_path = os.path.join(root, file_name)
                    print(f"Processing: {file_path}")
                    
                    # Loading audio file
                    y, sr = librosa.load(file_path)
                    # Generating mel spectogram
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    # convert to decible scale
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    
                    # creating class-specific folder in the output directory
                    relative_path = os.path.relpath(root, input_directory)
                    class_folder = os.path.join(output_directory, relative_path)
                    os.makedirs(class_folder, exist_ok=True)
                    
                    # Saving spectogram as an image
                    output_file = os.path.join(class_folder, f"{os.path.splitext(file_name)[0]}.png")
                    plt.figure(figsize=(2, 2))
                    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
                    plt.axis('off')
                    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    print(f"Saved spectrogram to: {output_file}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Function to load and process the data
def prepare_data(dataset_path):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        dataset_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        dataset_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation'
    )
    return train_gen, val_gen

# Function to build CNN-RNN model
def build_rnn_model(input_shape, num_classes):
    model = models.Sequential()

    # convolutional layers for feature extraction
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2, 2))

   
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))

    
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))

     # flatten the feature maps into a single vector
    model.add(layers.Flatten())

    # Determine the output shape after flattening 
    conv_output_shape = model.output_shape[1] 

    # defining LSTM input dimensions
    timesteps = 16 # number of time steps for LSTM
    reshape_dim = conv_output_shape // timesteps

    # Ensuring that the reshape operation does not alter tensor size
    if reshape_dim * timesteps != conv_output_shape:
        raise ValueError("Chosen timesteps do not evenly divide conv_output_shape.")

    # reshaping for LSTM processing 
    model.add(layers.Reshape((timesteps, reshape_dim)))  

    # LSTM layers for sequence learning
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(64))

    # fully connected layers for classification
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# function to train the model
def train_model():
    train_gen, val_gen = prepare_data(DATASET_PATH)
    model = build_rnn_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=train_gen.num_classes)

    print("Training the RNN model...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    
    # save the trained model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    return model, history, val_gen

# function to plot train model
def plot_history(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# function to evaluate trained model
def evaluate_model(model, val_gen):
    val_predictions = model.predict(val_gen)
    y_pred = np.argmax(val_predictions, axis=1)
    y_true = val_gen.classes

    # Computing confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # printing classification report 
    test_accuracy = model.evaluate(val_gen, verbose=0)
    print(f"Test Accuracy: {test_accuracy[1] * 100:.2f}%")

# main execution block
if __name__ == "__main__":
    audio_to_spectrogram("Notes Datasets", "specto1") # converting audio files to spectogram
    model, history, val_gen = train_model() # Training the model
    plot_history(history) # plot training accuracy over epochs
    evaluate_model(model, val_gen) # evaluating the model performance
