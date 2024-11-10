import os
import cv2
import supervision as sv
from ultralytics import YOLO
from utils import get_center_of_bbox
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from torchvision import models

import csv

class Tracker:
    def __init__(self, detection_model_path, classification_model_path, detection_conf=0.1, input_size=(640, 1280), fps=30):
        self.model = YOLO(detection_model_path)
        self.class_names = {0: "freestyle", 1: "start", 2: "underwater"}  # Added class 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classification_model = self.load_classification_model(classification_model_path).to(self.device)
        self.tracker = sv.ByteTrack(track_buffer=fps * 10, track_thresh=0.25, match_thresh=0.95, frame_rate=fps)
        self.transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.activity_segments = []
        self.detection_conf = detection_conf
        self.input_size = input_size
        self.fps =fps

    def load_classification_model(self, model_path):
        class MobileNetV2(nn.Module):
            def __init__(self, num_classes):
                super(MobileNetV2, self).__init__()
                self.model = models.mobilenet_v2(pretrained=False)
                self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)  # Adjust according to the new model's output features

            def forward(self, x):
                return self.model(x)

        model = MobileNetV2(num_classes=len(self.class_names))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def load_image(self, image_path, transform):
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)
        return image

    def predict(self, model, image_path, transform, device):
        model.eval()
        image = self.load_image(image_path, transform).to(device)
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return predicted.cpu().item(), confidence.cpu().item()

    def display_prediction(self, image_path, prediction, confidence, track_dir):
        image = Image.open(image_path)
        predicted_class = self.class_names[prediction]
        plt.imshow(image)
        plt.title(f'Prediction: {predicted_class} with confidence {confidence:.2f}')
        plt.axis('off')
        prediction_output_path = os.path.join(track_dir, f"prediction_{os.path.basename(image_path)}")
        plt.savefig(prediction_output_path)
        plt.close()

    def detect_frame(self, frame):
        results = self.model(frame, conf=self.detection_conf, imgsz=self.input_size)[0]
        return results

    def detect_frames(self, frames):
        swimmer_detections = []

        for frame in frames:
            detection = self.detect_frame(frame)
            swimmer_detections.append(detection)

        return swimmer_detections

    def classify_image(self, image):
        image_path = "temp_image.png"
        cv2.imwrite(image_path, image)
        prediction, confidence = self.predict(self.classification_model, image_path, self.transform, self.device)
        os.remove(image_path)
        return prediction, confidence

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

        tracks = {
            "swimmers": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = self.model.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["swimmers"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                detection_confidence = frame_detection[2]

                if cls_id == cls_names_inv.get('swimmer', None):
                    center_position = get_center_of_bbox(bbox)

                    tracks["swimmers"][frame_num][track_id] = {
                        "bbox": bbox,
                        "position": center_position,
                        "detection_confidence": detection_confidence
                    }

        return tracks

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])

        overlay = frame.copy()
        cv2.drawContours(overlay, [triangle_points], 0, color, cv2.FILLED)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        shadow_offset = 2
        shadow_points = np.array([
            [x + shadow_offset, y + shadow_offset],
            [x - 10 + shadow_offset, y - 20 + shadow_offset],
            [x + 10 + shadow_offset, y - 20 + shadow_offset],
        ])
        cv2.drawContours(frame, [shadow_points], 0, (50, 50, 50), cv2.FILLED)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.drawContours(frame, [shadow_points], 0, (0, 0, 0), 2)

        return frame
    
    def draw_annotations(self, video_frames, tracks, global_frame_index):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            swimmer_dict = tracks["swimmers"][frame_num]

            for track_id, swimmer in swimmer_dict.items():
                bbox = swimmer["bbox"]
                frame = self.draw_triangle(frame, bbox, (0, 255, 0))
                x, y = get_center_of_bbox(bbox)

                x1, y1, x2, y2 = map(int, bbox)
                bbox_image = frame[y1:y2, x1:x2]

                classification, classification_confidence = self.classify_image(bbox_image)
                class_name = self.class_names.get(classification, "unknown")  # Handle unknown classifications

                cv2.putText(frame, f"ID: {track_id} - Class: {class_name} - Class Conf: {classification_confidence:.2f}",
                            (x - 10, y - 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

                # Ensure segment is a dictionary
                segment = {
                    'frame_index': global_frame_index + frame_num,
                    'track_id': track_id,
                    'class': class_name,
                    'bbox': bbox,
                    'confidence': classification_confidence,
                    'start_time': (global_frame_index + frame_num) / self.fps,
                    'end_time': (global_frame_index + frame_num + 1) / self.fps
                }

                self.activity_segments.append(segment)

            output_video_frames.append(frame)

        return output_video_frames
    
    '''
    def draw_annotations(self, video_frames, tracks, global_frame_index):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            swimmer_dict = tracks["swimmers"][frame_num]

            for track_id, swimmer in swimmer_dict.items():
                bbox = swimmer["bbox"]
                frame = self.draw_triangle(frame, bbox, (0, 255, 0))
                x, y = get_center_of_bbox(bbox)

                x1, y1, x2, y2 = map(int, bbox)
                bbox_image = frame[y1:y2, x1:x2]

                classification, classification_confidence = self.classify_image(bbox_image)
                class_name = self.class_names.get(classification, "unknown")  # Handle unknown classifications

                cv2.putText(frame, f"ID: {track_id} - Class: {class_name} - Class Conf: {classification_confidence:.2f}",
                            (x - 10, y - 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                
                if track_id not in self.activity_segments:
                    self.activity_segments[track_id] = []

                if not self.activity_segments[track_id] or self.activity_segments[track_id][-1]['class'] != class_name:
                    if self.activity_segments[track_id]:
                        self.activity_segments[track_id][-1]['end_frame'] = global_frame_index + frame_num - 1

                    self.activity_segments[track_id].append({
                        'start_frame': global_frame_index + frame_num,
                        'class': class_name,
                        'bbox': bbox,
                        'end_frame': global_frame_index + frame_num
                    })
                else:
                    self.activity_segments[track_id][-1]['end_frame'] = global_frame_index + frame_num

            output_video_frames.append(frame)

        return output_video_frames
    '''
    '''
    def save_activity_segments_to_csv(self, frame_rate, csv_file_path):
        with open(csv_file_path, mode='w', newline='') as csv_file:
            fieldnames = ['Swimmer ID', 'Activity', 'Start Time (s)', 'End Time (s)', 'Duration (s)', 'BBox']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for track_id, segments in self.activity_segments.items():
                for segment in segments:
                    start_time = round(segment['start_frame'] / frame_rate, 2)
                    end_time = round(segment['end_frame'] / frame_rate, 2)
                    duration = round(end_time - start_time, 2)
                    bbox = segment['bbox']
                    writer.writerow({
                        'Swimmer ID': track_id,
                        'Activity': segment['class'].capitalize(),
                        'Start Time (s)': start_time,
                        'End Time (s)': end_time,
                        'Duration (s)': duration,
                        'BBox': bbox
                    })
    '''             
    def save_activity_segments_to_csv(self, frame_rate, csv_file_path):
        with open(csv_file_path, mode='w', newline='') as csv_file:
            fieldnames = ['Frame Index', 'Swimmer ID', 'Activity', 'BBox', 'Confidence', 'Start Time (s)', 'End Time (s)']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for segment in self.activity_segments:
                # Debugging: Ensure each segment is a dictionary and print its contents
                print(segment)
                if isinstance(segment, dict):
                    writer.writerow({
                        'Frame Index': segment['frame_index'],
                        'Swimmer ID': segment['track_id'],
                        'Activity': segment['class'].capitalize(),
                        'BBox': segment['bbox'],
                        'Confidence': segment['confidence'],
                        'Start Time (s)': segment['start_time'],
                        'End Time (s)': segment['end_time']
                    })
                else:
                    print(f"Invalid segment format: {segment}")
