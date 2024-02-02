# YOLOv8 Object Detection with OpenCV and Ultralytics

This Python script uses YOLOv8 from Ultralytics for real-time object detection using OpenCV. The script initializes a camera, loads the YOLOv8 model, and processes frames from the camera, annotating detected objects with bounding boxes.

## Prerequisites

Make sure you have the following libraries installed:

- `cv2` (OpenCV)
- `ultralytics`
- `supervision`
```bash
pip install opencv-python
pip install ultralytics
pip install supervision
```


## Getting Started
Clone the repository:
```bash
git clone https://github.com/SYED-M-HUSSAIN/YOLOv8-Object-Detection-with-OpenCV-and-Ultralytics
```
## Usage
Run the script:
```bash
python yolo_inference.py
```
## Configuration

    initialize_camera: Initializes the camera using OpenCV.
    load_yolov8_model: Loads the YOLOv8 model from Ultralytics.
    process_frame: Processes each frame from the camera using the YOLOv8 model and annotates the detected objects.
    main: The main function that captures frames from the camera, processes them, and displays the annotated frames.

## Customization

    You can customize the camera index, model path, and annotation parameters according to your needs.
