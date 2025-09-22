---
🧠 Face Mask Detection Web App

This project is a real-time face mask detection web application built with Flask, OpenCV, and MediaPipe. It uses a webcam to detect whether a person is wearing a mask based on facial landmarks and basic skin detection logic.


---

---
🚀 Features

Real-time mask detection via webcam

Uses MediaPipe FaceMesh for precise facial landmark tracking

Detects nose and mouth exposure using skin color analysis in the HSV color space

Stream video feed directly on a web page



---

🧰 Technologies Used

Python 3.x

Flask

OpenCV

MediaPipe

HTML (for rendering the stream)



---

📦 Installation

1. Clone the repository

git clone https://github.com/bharath20056/Face-mask-detection.git
cd face-mask-detection

2. Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

requirements.txt Example

flask
opencv-python
mediapipe
numpy


---

🧪 Running the Application

python app.py

Open your browser and go to:

http://127.0.0.1:5000/

You'll see the webcam stream with mask detection in action.
---
---
📌 Notes

This app uses a basic heuristic approach with HSV-based skin detection. For production or critical use, consider using a CNN or a pre-trained deep learning model for better accuracy.

Make sure your webcam is accessible and not being used by another application.

Performance may vary depending on lighting conditions and camera quality.



---

📖 Future Improvements

Add deep learning-based mask classification (e.g., MobileNet, YOLOv5)

Handle multiple faces simultaneously

Improve skin detection using adaptive thresholds

Add UI styling and control buttons



---
