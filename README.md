# Drowsiness Detection System

A deep learning-based computer vision system that detects drowsiness by analyzing facial cues such as eye closure and yawning. The model is deployed as an interactive web application using Gradio.

---

## Overview

This project classifies facial states into four categories:

* Closed Eyes
* Open Eyes
* Yawning
* No Yawn

It uses a MobileNetV2-based convolutional neural network trained on a custom dataset. The system processes input images and predicts the user’s state, with a counter-based mechanism to detect sustained drowsiness and trigger alerts.

---

## Features

* Real-time inference using webcam or uploaded images
* Deep learning model trained on facial state classification
* Drowsiness alert based on consecutive predictions
* Lightweight architecture suitable for real-time applications
* Deployed using Gradio on Hugging Face Spaces

---

## Model Details

* Architecture: MobileNetV2
* Input Size: 128 × 128
* Classes: 4 (Closed, Open, Yawn, No Yawn)
* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Evaluation Metrics: Accuracy, Confusion Matrix

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

Run the application locally:

```bash
python app.py
```

Then:

* Upload an image or use webcam input
* Capture a frame
* Click "Predict" to get the result

---

## Results

* Achieves high accuracy on test data
* Effectively distinguishes between eye closure and yawning states
* Demonstrates stable performance across different inputs

---

## Limitations

* Requires full-face input for best performance
* Webcam input is frame-based (not continuous streaming)
* Performance depends on lighting and image quality

---

## Future Improvements

* Real-time video stream processing
* Separate eye and mouth detection pipelines
* Audio alert integration
* Model optimization for edge devices

---

## License

This project is licensed under the MIT License.

---

## Author

Vishwanath SA
