import cv2
from tqdm import tqdm
import numpy as np

class VideoProcessor:
    """
    Handles video processing tasks like reading, skipping frames, and writing annotated frames.
    """
    def __init__(self, input_video_path, output_video_path, frame_skip=10):
        """
        Initialize the VideoProcessor with video paths and frame skip settings.

        Parameters:
            input_video_path (str): Path to the input video file.
            output_video_path (str): Path to save the annotated output video.
            frame_skip (int): Number of frames to skip between processing.
        """
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.frame_skip = frame_skip

        # Initialize video capture and writer
        self.cap = cv2.VideoCapture(input_video_path)
        if not self.cap.isOpened():
            raise ValueError("Error opening video file.")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.out = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
            self.fps,
            (self.width, self.height)
        )

    def process_frames(self, processors):
        """
        Process frames in the video using a list of processors and skip unnecessary frames.

        Parameters:
            processors (list): A list of callable frame processors.
        """
        frame_count = 0
        last_annotations = []  # Store annotations from the last processed frame

        for _ in tqdm(range(self.total_frames), desc="Processing video"):
            analyze = (frame_count % self.frame_skip == 0)

            ret, frame = self.cap.read()
            if not ret:
                print("Error reading video frame.")
                break

            if analyze:
                # Process the frame using all processors
                annotations = []
                for processor in processors:
                    frame, processor_annotations = processor.process_frame(frame, analyze=True)
                    annotations.extend(processor_annotations)
                last_annotations = annotations  # Update last annotations
            #else:
            # Draw the last annotations on skipped frames
            for annotation in last_annotations:
                x, y, w, h = annotation["region"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                   f"{annotation['name']} - {annotation['dominant_emotion']}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                     2
                )

            # Write the processed frame to the output video
            self.out.write(frame)
            frame_count += 1

        # Release resources
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
