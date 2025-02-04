# Guitar Note Classifier

## Overview
The Guitar Note Classifier is a machine learning project designed to classify guitar notes from audio files. The system processes `.wav` and `.mp3` files, detects note onsets, converts them into spectrogram images, and uses a trained Convolutional Neural Network (CNN) to classify these notes. This README explains how to set up, run, and use the project effectively.

## Features
- Detects onsets (note start times) in an audio file.
- Converts audio segments into Mel spectrograms for classification.
- Trains a CNN model to classify guitar notes.
- Uses a trained model to predict notes in an input musical piece.
- Provides visualization of the waveform with detected onsets.

## Prerequisites
Before running the project, ensure you have the following installed:

### Required Software
- **Python** 3.8 or later

### Required Python Libraries:
- `tensorflow`
- `librosa`
- `matplotlib`
- `numpy`
- `scikit-learn`

### Installation
Install the required libraries using:
```sh
pip install tensorflow librosa matplotlib numpy scikit-learn
```

## Setup

### 1. Clone the Repository
Clone the project to your local machine:
```sh
git clone <repository_url>
cd <repository_directory>
```

### 2. Directory Structure
Organize your dataset as follows:
```
Guitar Dataset/
├── A2/
│   ├── file1.wav
│   ├── file2.wav
├── B3/
│   ├── file3.wav
│   ├── file4.wav
...
```
This structure ensures the script processes files from `Guitar Dataset` for training and testing.

## Model Training

### 1. Preprocessing & Training the CNN Model
Run the following script to train the model:
```sh
python train_model.py
```
This will:
- Load `.wav` files from `Guitar Dataset`.
- Convert each file into a **Mel spectrogram**.
- Train a CNN model to classify guitar notes.
- Save the trained model as `note_classifier.h5`.

### 2. Model Configuration
You can adjust model parameters in `train_model.py`:
- `TARGET_SHAPE`: Spectrogram image dimensions (default: `(128, 128)`).
- `EPOCHS`: Number of training epochs (default: `20`).
- `BATCH_SIZE`: Number of samples per batch (default: `32`).

## Note Detection in a Musical Piece
After training, you can use the model to detect notes in a musical file.

### 1. Run the Note Detection Script
```sh
python detect_notes.py
```
This script will:
- Load an input musical piece (`.mp3` or `.wav`).
- Detect note onsets using **Librosa's onset detection**.
- Extract audio segments around detected onsets.
- Convert each segment into a **Mel spectrogram**.
- Use the trained CNN model to classify the notes.
- Display detected notes and visualize the waveform.

### 2. Adjusting Parameters
Modify the following parameters in `detect_notes.py`:
- `MUSICAL_PIECE_PATH`: Path to the audio file.
- `SEGMENT_DURATION`: Duration of each note segment (default: `0.5` seconds).
- `SR`: Sampling rate (default: `44100`).

## Outputs

### 1. Trained Model
- Saved as `note_classifier.h5`.

### 2. Evaluation Metrics
- Accuracy and loss plots.
- Confusion matrix and classification report.

### 3. Detected Notes
- Displayed in the terminal as:
```
Time: 1.23s -> Note: A3
Time: 2.45s -> Note: Csharp4
```
- Waveform visualization with marked detected onsets.

## Troubleshooting

### No Notes Detected
- Ensure the input audio file has clear note attacks.
- Adjust onset detection parameters in `detect_notes.py`.

### Training Issues
- Verify dataset structure (`Guitar Dataset` should contain `.wav` files in subdirectories).
- Ensure enough samples exist for each class to prevent imbalance.

## Future Improvements
- Support for other audio formats like `.flac`.
- Implement a more advanced CNN architecture (e.g., **ResNet**).
- Develop a user-friendly GUI for real-time classification.

## Conclusion
This project provides an end-to-end solution for detecting and classifying guitar notes from audio files. By using onset detection and CNN-based classification, the system effectively identifies notes in a musical piece. Contributions and improvements are welcome!

