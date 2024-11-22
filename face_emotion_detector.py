import cv2
import face_recognition
from deepface import DeepFace
import numpy as np

class FaceEmotionDetector:
    """
    Detects faces, analyzes emotions, and annotates frames.
    """
    def __init__(self, known_face_encodings, known_face_names):
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.last_annotations = []  # Store annotations from the last analyzed frame

    def process_frame(self, frame, analyze=True):
        """
        Processes a single frame for face detection, emotion analysis, and annotation.

        Parameters:
            frame (numpy.ndarray): The input video frame.
            analyze (bool): Whether to analyze the frame or reuse previous annotations.

        Returns:
            tuple: Annotated frame and a list of annotations (bounding boxes and labels).
        """
        annotations = []

        if analyze:
            # Perform face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances) if matches else None

                if best_match_index is not None and matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                face_names.append(name)

            # Analyze emotions using DeepFace
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

            # Generate annotations
            for location, name in zip(face_locations, face_names):
                top, right, bottom, left = location
                annotations.append({
                    "region": (left, top, right - left, bottom - top),
                    "name": name,
                    "dominant_emotion": result[0]["dominant_emotion"] if result else "N/A"
                })

            # Update last annotations
            self.last_annotations = annotations
        else:
            # Reuse last annotations for skipped frames
            annotations = self.last_annotations

        # Annotate frame with bounding boxes
        # for annotation in annotations:
        #     x, y, w, h = annotation["region"]
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     cv2.putText(
        #         frame,
        #         f"{annotation['name']} - {annotation['dominant_emotion']}",
        #         (x, y - 10),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.9,
        #         (36, 255, 12),
        #         2
        #     )

        return frame, annotations
