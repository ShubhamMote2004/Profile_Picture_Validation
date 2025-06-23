import cv2
from deepface import DeepFace
from nudenet import NudeClassifier
from PIL import Image
import numpy as np

# Initialize NSFW classifier once
nsfw_classifier = NudeClassifier()

def count_faces(image_path):
    """Count number of faces using OpenCV Haar Cascade."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces)

def check_image_safety(image_path):
    """Check if image is safe using NudeNet."""
    result = nsfw_classifier.classify(image_path)
    scores = result.get(image_path, {})
    return scores.get("unsafe", 1) < 0.3  # Adjust threshold if needed

def verify_profile_image(image_path, expected_gender=None):
    """Main function to verify a profile image (gender check skipped)."""
    try:
        # 1. Face count (also serves as human detection)
        face_count = count_faces(image_path)
        is_single_person = face_count == 1
        human_face_detected = face_count > 0

        # 2. NSFW check
        is_safe = check_image_safety(image_path)

        # Final output (gender prediction skipped)
        result = {
            "✅ Status": "Success",
            # "🧠 Detected Gender": f"{predicted_label} ({predicted_score:.2f}%)",  # Removed
            # "🎯 Gender Match": gender_match,  # Removed
            "🧍 Single Person = ": is_single_person,
            "🧑‍💻 Human Face Detected = ": human_face_detected,
            "🔒 Is it Safe Image? = ": is_safe,
            "🧑 Total Faces Detected = ": face_count
        }
        return result

    except Exception as e:
        return {
            "❌ Status": "Failed",
            "Error": str(e),
            "🧑‍💻 Human Face Detected = ": False,
            "🧍 Single Person = ": False,
            "🔒 Is it Safe Image? = ": False
        }

if __name__ == "__main__":
    image_path = "test_images/sample_image_nsfw.jpg"
    # Gender not needed anymore
    result = verify_profile_image(image_path)

    print("\n📷 Profile Image Verification Result:")
    for key, value in result.items():
        print(f"{key}: {value}")
