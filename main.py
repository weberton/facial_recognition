from face_loader import FaceLoader
from face_emotion_detector import FaceEmotionDetector
from video_processor import VideoProcessor
from pose_landmark_detector import PoseLandmarkDetector
import os

if __name__ == "__main__":
    # Load known faces
    image_folder = "images"
    face_loader = FaceLoader(image_folder)
    face_loader.load_faces()
    known_face_encodings, known_face_names = face_loader.get_known_faces()

    # Input and output video paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(script_dir, 'videos/challenge.mp4')
    output_video_path = os.path.join(script_dir, 'videos/challenge_output.mp4')

  # Initialize processors
    face_emotion_detector = FaceEmotionDetector(known_face_encodings, known_face_names)
    pose_landmark_detector = PoseLandmarkDetector()

    video_processor = VideoProcessor(input_video_path, output_video_path, frame_skip=10)

    # Process video
    video_processor.process_frames([face_emotion_detector, pose_landmark_detector])
    #video_processor.process_frames([face_emotion_detector])