import csv
import os
import logging
import psutil
import gc
from ultralytics import YOLO
from utils import read_video_in_chunks, save_video_incremental, get_video_fps#, detect_lane_ropes
from trackers import Tracker
from excel_helpers import calculate_average_speed

logging.basicConfig(level=logging.INFO)

def process_batch(frames_batch, tracker, output_path, batch_index, global_frame_index, fps):
    # Get object tracks for the current batch of frames
    tracks = tracker.get_object_tracks(frames_batch)
    # Draw annotations on the frames based on the tracks
    annotated_frames = tracker.draw_annotations(frames_batch, tracks, global_frame_index)
    # Save the annotated frames incrementally to the output video file
    save_video_incremental(annotated_frames, output_path, batch_index, fps)
    return len(frames_batch)  # Return the number of frames processed in this batch

def main():
    video_path = 'test2.mp4'  # Path to the input video file
    output_path = 'output_videos/output.mp4'  # Path to the output video file
    detection_model_path = 'models/best (17).pt'  # Path to the detection model
    classification_model_path = 'models/final_model_weights (14).pth'  # Path to the classification model
    detection_conf = 0.3  # Confidence threshold for the detection model
    input_size = (384, 640)  # Desired input size for the model

    # Get the fps of the input video
    fps = get_video_fps(video_path)

    tracker = Tracker(detection_model_path, classification_model_path, detection_conf, input_size, fps)
    batch_size = 500  # Batch size for processing video chunks

    # Remove previous output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    batch_index = 0
    global_frame_index = 0

    # Process video in chunks
    for frames_batch in read_video_in_chunks(video_path, batch_size):
        frames_processed = process_batch(frames_batch, tracker, output_path, batch_index, global_frame_index, fps)
        global_frame_index += frames_processed
        batch_index += 1

        # Log memory usage
        memory_info = psutil.virtual_memory()
        logging.info(f"Memory usage: {memory_info.percent}% used, {memory_info.available / (1024 * 1024):.2f} MB available")
        logging.info(f"Processed batch {batch_index} with {len(frames_batch)} frames.")
        
        # Force garbage collection to free memory
        gc.collect()
        
    # Save activity segments to CSV file
    tracker.save_activity_segments_to_csv(frame_rate=fps, csv_file_path='stats.csv')
    # Calculate average speed from the saved CSV file
    calculate_average_speed('stats.csv')

if __name__ == '__main__':
    main()
