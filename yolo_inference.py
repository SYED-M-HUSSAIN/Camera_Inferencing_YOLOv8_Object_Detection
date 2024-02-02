import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    return cap

def load_yolov8_model(model_path="yolov8n.pt"):
    model = YOLO(model_path)
    return model

def process_frame(frame, model, box_annotator):
    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in detections
    ]
    annotated_frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )
    return annotated_frame

def main():
    cap = initialize_camera()
    model = load_yolov8_model()
    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        annotated_frame = process_frame(frame, model, box_annotator)

        cv2.imshow("yolov8", annotated_frame)

        if cv2.waitKey(20) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
