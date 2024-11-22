import cv2
import mediapipe as mp

class PoseLandmarkDetector:
    """
    Detects human pose landmarks using Mediapipe.
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_landmarks = None  # Store landmarks from the last analyzed frame

    def process_frame(self, frame, analyze=True):
        """
        Processes a single frame for pose detection and annotation.

        Parameters:
            frame (numpy.ndarray): The input video frame.
            analyze (bool): Whether to analyze the frame or reuse previous landmarks.

        Returns:
            tuple: Annotated frame and an empty list (no bounding box annotations).
        """
        if analyze:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                # Update last landmarks with the current results
                self.last_landmarks = results.pose_landmarks

        # Draw the last landmarks on the frame
        if self.last_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, self.last_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

        return frame, []  # Pose detection doesn't return bounding boxes
