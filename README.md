# Emotion Recognition System Based on Voice and Facial Expressions

This project proposes a multi-modal emotion recognition system that integrates visual facial expressions and acoustic features. Designed to successfully distinguish four main emotions (neutral, happiness, sadness, and anger), it determines the subject's current emotional state by analyzing dynamic changes in facial expressions and acoustic waveform patterns.

## Contributors

* NTHU EE26 111011114 張允澤
* NTHU EE26 111061147 李羿軒
* NTHU EE26 111061217 曾介績

## System Architecture

The system operates on a synchronized state-machine architecture comprising a calibration phase and a recognition phase. Unlike traditional deep learning approaches, it employs a baseline-calibration mechanism to address individual differences in facial structure and vocal characteristics.

* **Visual Module**: Utilizes a hybrid approach combining the texture analysis of DeepFace with the precise geometric tracking of MediaPipe Face Mesh. All geometric measurements, such as the Brow-Eye Compression Ratio and Mouth Frown Ratio, are normalized by the Face Height to ensure robustness against camera distance.
* **Audio Module**: Extracts 13 acoustic features using the Librosa library, including Pitch, Energy, and Spectral Centroid. It employs a Psychoacoustic Heuristic Classifier to explicitly codify acoustical fingerprints into the decision logic.
* **Multimodal Fusion**: Integrates signals using a Weighted Late Fusion strategy, assigning equal weights (0.5) to visual and audio modalities. To prevent unstable flipping between low-confidence states, a Temperature-Scaled Softmax function (Temperature = 0.8) is applied, treating "Neutral" as an active class that competes directly with other emotions.

## Dependencies

Ensure you have Python installed along with the required packages:

* `numpy >= 1.21.0`
* `opencv-python >= 4.5.0`
* `mediapipe >= 0.10.0`
* `deepface >= 0.0.79`
* `librosa >= 0.10.0`
* `sounddevice >= 0.4.6`
* `tensorflow >= 2.15.0`

## How to Run

1. **Start the System**: Execute `main.py`.
2. **Step 1: Calibration**: Stay neutral and silent for 5 seconds to establish personalized visual and audio baselines.
3. **Step 2: Ready**: Wait for the system prompt.
4. **Step 3: Recording**: Press Enter to record a 5-second window. The system will compute the relative deviations from your baseline and output the fused predicted emotion.
