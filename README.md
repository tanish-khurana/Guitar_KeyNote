# README: Guitar Note Classifier

## Overview
The Guitar Note Classifier is a machine learning project that classifies guitar notes using audio files. The system processes `.wav` files, converts them into spectrogram images, and trains a Convolutional Neural Network (CNN) to classify these notes. This README explains how to set up, run, and use the project effectively.

---

## Features
- Converts audio files in `.wav` format into spectrogram images.
- Handles nested directory structures for audio files.
- Trains a CNN to classify guitar notes from spectrograms.
- Provides evaluation metrics like accuracy, confusion matrix, and classification report.

---

## Prerequisites
Before running the project, ensure you have the following installed:

- Python 3.8 or later
- Required Python libraries:
  - `tensorflow`
  - `librosa`
  - `matplotlib`
  - `numpy`
  - `seaborn`
  - `scikit-learn`

Install the required libraries using:
```bash
pip install tensorflow librosa matplotlib numpy seaborn scikit-learn
```

---

## Setup
### 1. Clone the Repository
Clone the project to your local machine:
```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Directory Structure
Organize your audio files in a directory. The structure should look like this:
```
Notes Datasets/
├── Class1/
│   ├── file1.wav
│   ├── file2.wav
├── Class2/
│   ├── file3.wav
│   ├── file4.wav
```

The script will process files from the `Notes Datasets` directory and save spectrograms in the `specto1` directory.

### 3. Adjust Parameters
Open the script file (`keynotes.py`) and adjust parameters if needed:
- `IMAGE_SIZE`: Dimensions of spectrogram images (default: `(128, 128)`).
- `BATCH_SIZE`: Batch size for training (default: `32`).
- `EPOCHS`: Number of training epochs (default: `30`).
- `DATASET_PATH`: Path to the spectrogram directory (`specto1`).
- `MODEL_SAVE_PATH`: Path to save the trained model (`guitar_note_classifier.h5`).

---

## How to Run the Project
### 1. Generate Spectrograms
Run the script to process all `.wav` files in the `Notes Datasets` directory and generate spectrograms:
```bash
python keynotes.py
```

This will:
1. Process audio files recursively.
2. Save spectrograms in `specto1` with the same folder structure as `Notes Datasets`.

### 2. Train the Model
The script will automatically start training the CNN once spectrograms are generated. The model will be saved to the path specified in `MODEL_SAVE_PATH`.

### 3. Evaluate the Model
After training, the script will:
- Plot training and validation accuracy over epochs.
- Generate a confusion matrix and classification report.

---

## Customization
### Add More Classes
To train the model with additional classes:
1. Add new folders to `Notes Datasets`, each representing a class.
2. Place `.wav` files in the appropriate folders.
3. Re-run the script to regenerate spectrograms and retrain the model.

### Use a Pre-trained Model
If you have a pre-trained model, you can load it using:
```python
from tensorflow.keras.models import load_model
model = load_model('path_to_saved_model.h5')
```

---

## Outputs
1. **Trained Model**:
   - Saved to the path specified in `MODEL_SAVE_PATH`.
2. **Evaluation Metrics**:
   - Accuracy and loss plots.
   - Confusion matrix and classification report.
3. **Spectrograms**:
   - Saved in `specto1`, organized by class.

---

## Troubleshooting
### No Images Found
- Ensure that `Notes Datasets` contains `.wav` files in correctly named subdirectories.
- Verify that the spectrogram generation step ran successfully.

### Training Issues
- Check that `specto1` contains spectrograms in the correct structure.
- Ensure there are enough samples for each class to avoid imbalance.

---

## Future Improvements
1. Add support for other audio formats like `.mp3` or `.flac`.
2. Implement advanced CNN architectures like ResNet.
3. Develop a user-friendly interface for real-time classification.

---
