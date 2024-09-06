#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

class YOLOv8Node(Node):
    def __init__(self):
        super().__init__('yolov8_node')
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/yolov8/annotated_image', 10)
        self.cap = cv2.VideoCapture(0)
        self.model = self.load_yolov8_model()
        self.box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)  # 30Hz

    def load_yolov8_model(self, model_path="yolov8n.pt"):
        model = YOLO(model_path)
        return model

    def process_frame(self, frame):
        result = self.model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{self.model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detections
        ]
        annotated_frame = self.box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )
        return annotated_frame

    def timer_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().error("Failed to capture frame")
            return

        annotated_frame = self.process_frame(frame)

        try:
            image_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.image_pub.publish(image_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")

def main(args=None):
    rclpy.init(args=args)
    yolov8_node = YOLOv8Node()

    try:
        rclpy.spin(yolov8_node)
    except KeyboardInterrupt:
        pass
    finally:
        yolov8_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
