#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

class YOLOv8Node:
    def __init__(self):
        rospy.init_node('yolov8_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/yolov8/annotated_image", Image, queue_size=10)
        self.cap = cv2.VideoCapture(0)
        self.model = self.load_yolov8_model()
        self.box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)

    def load_yolov8_model(self, model_path="yolov8n.pt"):
        model = YOLO(model_path)
        return model

    def process_frame(self, frame):
        result = self.model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{self.model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        annotated_frame = self.box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )
        return annotated_frame

    def run(self):
        rate = rospy.Rate(30)  # 30Hz
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()

            if not ret:
                rospy.logerr("Failed to capture frame")
                break

            annotated_frame = self.process_frame(frame)

            try:
                image_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
                self.image_pub.publish(image_msg)
            except CvBridgeError as e:
                rospy.logerr(e)

            rate.sleep()

if __name__ == '__main__':
    try:
        node = YOLOv8Node()
        node.run()
    except rospy.ROSInterruptException:
        pass
