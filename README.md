<h1>Emotion Recognition Using MobileNetV2</h1>
This project implements a real-time facial emotion recognition system using a pre-trained MobileNetV2 model fine-tuned on a custom dataset. The system detects faces in a video feed (e.g., from a webcam) and classifies the emotions into one of seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The project leverages TensorFlow/Keras for model training and OpenCV for real-time face detection and visualization.

<h3>Project Overview</h3>
The goal of this project is to build an emotion recognition system capable of identifying human emotions from facial expressions in real-time. The system consists of two main components:

<ol>
<li>Model Training: A convolutional neural network (CNN) based on MobileNetV2 is fine-tuned using a dataset of labeled facial images.</li>
<li>Real-Time Detection: The trained model is deployed with OpenCV to detect faces and predict emotions from a live video feed.</li>
</ol>

<h3>Requirements</h3>
Python 3.6+
TensorFlow 2.x
OpenCV (cv2)
NumPy
Matplotlib (optional)


<h4><i>Install the required dependencies:</i></h4>
<p style={color:blue;} >pip install -r requirements.txt</p>