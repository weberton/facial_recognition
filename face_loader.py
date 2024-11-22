import face_recognition
import os

class FaceLoader:
    """
    Loads known faces and their encodings from a folder.
    """
    def __init__(self, folder):
        self.folder = folder
        self.known_face_encodings = []
        self.known_face_names = []

    def load_faces(self):
        """
        Load face encodings and names from images in the folder.
        """
        for filename in os.listdir(self.folder):
            if filename.endswith(('.jpg', '.jpeg')):
                image_path = os.path.join(self.folder, filename)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    face_encoding = face_encodings[0]
                    name = os.path.splitext(filename)[0][:-1]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)

    def get_known_faces(self):
        """
        Returns the loaded face encodings and names.
        """
        return self.known_face_encodings, self.known_face_names