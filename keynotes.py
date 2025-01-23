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


IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
output_image = "specto1"
input_audio = "Notes Datasets"
DATASET_PATH = "specto1"
MODEL_SAVE_PATH = "guitar_note_classifier.h5"


def audio_to_spectrogram(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for root, _, files in os.walk(input_directory):  # os.walk for recursive traversal
        for file_name in files:
            if file_name.endswith('.wav'):  # Process only .wav files
                try:
                    file_path = os.path.join(root, file_name)
                    print(f"Processing: {file_path}")

                    # Load the audio file
                    y, sr = librosa.load(file_path)
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    S_dB = librosa.power_to_db(S, ref=np.max)

                    # Create output directory structure
                    relative_path = os.path.relpath(root, input_directory)
                    class_folder = os.path.join(output_directory, relative_path)
                    os.makedirs(class_folder, exist_ok=True)

                    # Save the spectrogram
                    output_file = os.path.join(class_folder, f"{os.path.splitext(file_name)[0]}.png")
                    plt.figure(figsize=(2, 2))
                    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
                    plt.axis('off')
                    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    print(f"Saved spectrogram to: {output_file}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")



def prepare_data(dataset_path):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        dataset_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        dataset_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation'
    )
    return train_gen, val_gen

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    train_gen, val_gen = prepare_data(DATASET_PATH)
    model = build_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=train_gen.num_classes)

    print("Training the model...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    return model, history, val_gen

def plot_history(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def evaluate_model(model, val_gen):
    val_predictions = model.predict(val_gen)
    y_pred = np.argmax(val_predictions, axis=1)
    y_true = val_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))

if __name__ == "__main__":
    audio_to_spectrogram("Notes Datasets", "specto1")
    model, history, val_gen = train_model()
    plot_history(history)
    evaluate_model(model, val_gen)

