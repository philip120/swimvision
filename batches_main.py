from ultralytics import YOLO
from utils import read_video_in_chunks, save_video_incremental
from trackers import Tracker
import logging
import psutil
import gc
import os

logging.basicConfig(level=logging.INFO)

def process_batch(frames_batch, tracker, output_path, batch_index):
    # Initialize homography with the first frame of each batch
    if len(frames_batch) > 0:
        success = tracker.pool_mapper.update_homography(frames_batch[0])
        if not success:
            logging.warning("Failed to update homography matrix. Pool mapping may be inaccurate.")
    
    tracks = tracker.get_object_tracks(frames_batch)
    annotated_frames = tracker.draw_annotations(frames_batch, tracks)
    save_video_incremental(annotated_frames, output_path, batch_index)
    return annotated_frames

def main():
    video_path = 'Untitled video - Made with Clipchamp (10).mp4'
    output_path = 'output_videos/output.mp4'
    detection_model_path = 'models/best (17).pt'
    classification_model_path = 'models/final_model_weights (14).pth'
    keypoint_model_path = 'models/pool_map.pt'
    tracker = Tracker(detection_model_path, classification_model_path,keypoint_model_path)
    batch_size = 500  # Further reduce batch size to 10 frames

    # Remove previous output file if exists
    if os.path.exists(output_path):
        os.remove(output_path)

    batch_index = 0
    for frames_batch in read_video_in_chunks(video_path, batch_size):
        process_batch(frames_batch, tracker, output_path, batch_index)
        batch_index += 1
        
        # Log memory usage
        memory_info = psutil.virtual_memory()
        logging.info(f"Memory usage: {memory_info.percent}% used, {memory_info.available / (1024 * 1024):.2f} MB available")
        logging.info(f"Processed batch {batch_index} with {len(frames_batch)} frames.")
        
        # Force garbage collection to free memory
        gc.collect()

if __name__ == '__main__':
    main()
