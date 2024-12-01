import numpy as np
import cv2
from ultralytics import YOLO

class PoolMapper:
    def __init__(self, keypoint_model_path):
        # Load YOLOv8 pose model
        self.keypoint_model = YOLO(keypoint_model_path)
        
        # Create a simplified 2D pool representation
        self.pool_width = 1000  # pixels
        self.pool_height = 500  # pixels
        self.pool_lanes = 8
        self.pool_sections = 5  # 5 sections for 25 meters (5m each)
        
        # Initialize homography matrix as None
        self.homography_matrix = None
        
        self.speed_window_size = 30  # Number of frames for speed averaging
        
    def detect_pool_keypoints(self, frame):
        """Detect pool keypoints using YOLOv8-pose"""
        results = self.keypoint_model(frame)[0]
        keypoints = results.keypoints.data
        
        if len(keypoints) == 0:
            raise ValueError("No pool keypoints detected in frame")
            
        # Get the first detection's keypoints
        keypoints = keypoints[0].cpu().numpy()  # Shape: [num_keypoints, 3] (x, y, confidence)
        
        # Filter out low confidence detections
        min_confidence = 0.3
        valid_keypoints = []
        
        for kp in keypoints:
            if kp[2] > min_confidence:  # Check confidence
                valid_keypoints.append([kp[0], kp[1]])  # Only take x,y coordinates
        
        if len(valid_keypoints) < 4:
            raise ValueError(f"Not enough high-confidence keypoints detected. Found {len(valid_keypoints)}")
        
        # Convert to numpy array
        points = np.array(valid_keypoints[:4])
        
        # Sort points to ensure consistent ordering:
        # [bottom-left, bottom-right, top-left, top-right]
        # Sort by y-coordinate (vertical position) first
        y_sorted = points[points[:, 1].argsort()]
        top_points = y_sorted[:2]
        bottom_points = y_sorted[2:]
        
        # Sort top and bottom points by x-coordinate
        top_left = top_points[top_points[:, 0].argsort()][0]
        top_right = top_points[top_points[:, 0].argsort()][1]
        bottom_left = bottom_points[bottom_points[:, 0].argsort()][0]
        bottom_right = bottom_points[bottom_points[:, 0].argsort()][1]
        
        # Return ordered points
        corner_points = np.float32([
            bottom_left,
            bottom_right,
            top_left,
            top_right
        ])
        
        return corner_points
    
    def update_homography(self, frame):
        """Update homography matrix based on detected keypoints"""
        try:
            corner_points = self.detect_pool_keypoints(frame)
            
            # Define corresponding points in 2D pool representation
            pool_points = np.float32([
                [0, self.pool_height],              # Bottom left
                [self.pool_width, self.pool_height], # Bottom right
                [0, 0],                             # Top left
                [self.pool_width, 0]                # Top right
            ])
            
            # Calculate homography
            self.homography_matrix = cv2.getPerspectiveTransform(
                corner_points, 
                pool_points
            )
            return True
            
        except Exception as e:
            print(f"Failed to update homography: {str(e)}")
            return False
    
    def map_position(self, video_position):
        """Convert position from video coordinates to 2D pool coordinates"""
        if self.homography_matrix is None:
            return (self.pool_width // 2, self.pool_height // 2)  # Center of pool
        
        position = np.array([[video_position]], dtype='float32')
        transformed = cv2.perspectiveTransform(position, self.homography_matrix)
        mapped_pos = transformed[0][0]
        
        # Constrain y position to nearest lane
        lane_height = self.pool_height / self.pool_lanes
        lane_index = int(mapped_pos[1] / lane_height)
        lane_index = max(0, min(lane_index, self.pool_lanes - 1))
        
        # Center y position within the lane
        y_position = (lane_index + 0.5) * lane_height
        
        # Keep x position but constrain to pool bounds
        x_position = max(0, min(mapped_pos[0], self.pool_width-1))
        
        return (x_position, y_position)
    
    def draw_pool(self):
        """Create a 2D visualization of the pool with enhanced aesthetics"""
        # Create pool background with light blue color
        pool_img = np.ones((self.pool_height, self.pool_width, 3), dtype=np.uint8)
        pool_img[:, :] = (230, 216, 173)  # Light blue in BGR
        
        # Draw lane markers
        lane_height = self.pool_height / self.pool_lanes
        
        # Draw alternating lane backgrounds for better visibility
        for i in range(self.pool_lanes):
            y1 = int(i * lane_height)
            y2 = int((i + 1) * lane_height)
            if i % 2 == 0:
                pool_img[y1:y2, :] = (230, 206, 153)  # Slightly darker blue for even lanes
        
        # Draw lane ropes with dashed lines
        for i in range(self.pool_lanes + 1):
            y = int(i * lane_height)
            # Draw dashed white and red segments
            for x in range(0, self.pool_width, 30):
                start_x = x
                end_x = min(x + 15, self.pool_width)
                if i > 0 and i < self.pool_lanes:  # Don't draw on pool edges
                    # Draw white segment
                    cv2.line(pool_img, (start_x, y), (end_x, y), (255, 255, 255), 2)
                    # Draw red segment
                    if end_x + 15 <= self.pool_width:
                        cv2.line(pool_img, (end_x, y), (end_x + 15, y), (0, 0, 255), 2)
        
        # Draw vertical distance markers (5m intervals)
        section_width = self.pool_width / self.pool_sections
        for i in range(self.pool_sections + 1):
            x = int(i * section_width)
            # Draw dotted line
            for y in range(0, self.pool_height, 20):
                cv2.line(pool_img, (x, y), (x, min(y + 10, self.pool_height)), (0, 0, 0), 1)
            
            # Add distance markers with better visibility
            distance = i * 5  # 5m intervals
            cv2.putText(pool_img, f"{distance}m", 
                       (x - 20, self.pool_height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw lane numbers with better visibility
        for i in range(self.pool_lanes):
            lane_number = i + 1
            y = int((i + 0.5) * lane_height)
            # Draw white background for text
            text_size = cv2.getTextSize(f"Lane {lane_number}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(pool_img, 
                         (5, y - text_size[1] - 5),
                         (5 + text_size[0] + 10, y + 5),
                         (255, 255, 255),
                         -1)
            # Draw text
            cv2.putText(pool_img, f"Lane {lane_number}", 
                       (10, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw pool borders with thicker lines
        cv2.rectangle(pool_img, (0, 0), (self.pool_width-1, self.pool_height-1), (0, 0, 0), 3)
        
        return pool_img
    
    def calculate_speed(self, position1, position2, time_interval=1/30):
        """
        Calculate instantaneous speed in meters per second
        """
        # Convert pixel distances to meters
        pixels_per_meter = self.pool_width / 25  # 25 meters pool length
        
        # Calculate distance in pixels
        dx = position2[0] - position1[0]
        dy = position2[1] - position1[1]
        distance_pixels = np.sqrt(dx**2 + dy**2)
        
        # Convert to meters
        distance_meters = distance_pixels / pixels_per_meter
        
        # Calculate speed
        speed = distance_meters / time_interval if time_interval > 0 else 0
        
        return speed
    
    def draw_swimmer(self, pool_img, position, track_id=None, activity_class=None, speed=None):
        """Draw a swimmer on the 2D pool visualization with enhanced visuals"""
        x, y = position
        
        # Ensure coordinates are within bounds
        x = max(0, min(int(x), self.pool_width-1))
        y = max(0, min(int(y), self.pool_height-1))
        
        # Calculate lane number (1-based)
        lane_height = self.pool_height / self.pool_lanes
        lane_number = int(y / lane_height) + 1
        
        # Draw swimmer marker with better visibility
        # Draw white outline
        cv2.circle(pool_img, (x, y), 7, (255, 255, 255), -1)
        # Draw colored center
        cv2.circle(pool_img, (x, y), 5, (0, 0, 255), -1)  # Red center
        
        # Create label with all available information
        label_parts = []
        if track_id is not None:
            label_parts.append(f"ID:{track_id}")
        label_parts.append(f"L{lane_number}")
        if activity_class is not None:
            label_parts.append(activity_class)
        if speed is not None:
            label_parts.append(f"{speed:.1f}m/s")
        
        label = " ".join(label_parts)
        
        # Add label with better visibility
        # Draw white background for text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(pool_img, 
                     (x + 5, y - text_size[1] - 5),
                     (x + text_size[0] + 15, y + 5),
                     (255, 255, 255),
                     -1)
        # Draw text
        cv2.putText(pool_img, label, (x + 10, y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return pool_img