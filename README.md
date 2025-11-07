# 🩷 Doll Detection with YOLOv8

This project uses **Roboflow** to train a **custom YOLOv8 model** for detecting dolls in images and videos.
It includes a full pipeline for running inference with your trained weights (`best.pt`).

---

##  Setup Instructions

### 1️ Create a Virtual Environment

```bash
python -m venv env_doll
```

### 2️ Activate the Environment

**Windows**

```bash
env_doll\Scripts\activate
```

**Mac / Linux**

```bash
source env_doll/bin/activate
```

---

##  Install Dependencies

Make sure your virtual environment is active, then install all required libraries:

```bash
pip install -r requirements.txt
```

This will install:

* **Ultralytics** (YOLOv8 framework)
* **Roboflow** (for dataset management)
* **PyTorch** (deep learning backend)
* **OpenCV** and **Pillow** (for image processing)
* **Matplotlib** and **NumPy** (for visualization and numerical tasks)

---

##  Run Detection

After installing dependencies, run your detection script:

```bash
python detection.py
```

Make sure your YOLO weights file (`best.pt`) is in the same folder as `detection.py`.

The script will:

1. Load your trained YOLO model
2. Perform object detection on the input image or video
3. Display and save the annotated results

---

##  Notes

* You can modify the image or video source path directly in `detection.py`.
* To use a webcam, change the source to `0` inside the script.
* Ensure that **your environment’s Python version is compatible** with your installed `torch` and `numpy` versions (e.g., Python 3.12+).

---

##  Example Project Structure

```
Doll_Detection_YOLO/
│
├── detection.py
├── best.pt
├── requirements.txt
├── env_doll/
└── test_image.jpg
```

