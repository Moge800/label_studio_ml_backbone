# yolo_backend.py

from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
from PIL import Image
import requests
import os
import uuid
import hidden_key as KEY

# Configuration
# Replace with your Label Studio host if different
# Example: LABEL_STUDIO_HOST = "http://your-label-studio-host.com:<port>"
LABEL_STUDIO_HOST = KEY.LABEL_STUDIO_HOST

# Replace with your Label Studio API key
LABEL_STUDIO_API_KEY = KEY.LABEL_STUDIO_API_KEY
# Path to your YOLO model
YOLO_MODEL_PATH = KEY.MODEL_PATH

# Class ID to label name mapping based on user's label config
# Example usage:
# CLASS_NAMES={0:"<label_name1>",1:"<label_name2>",...}
CLASS_NAMES = KEY.CLASS_NAMES


def download_image(url, save_path):
    """Download image from Label Studio with authentication."""
    if url.startswith("/"):
        url = LABEL_STUDIO_HOST + url
    headers = {"Authorization": f"Token {LABEL_STUDIO_API_KEY}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)


class YOLOv11nModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = YOLO(YOLO_MODEL_PATH)

    def predict(self, tasks, **kwargs):
        predictions = []

        for task in tasks:
            image_url = task["data"]["image"]
            temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
            download_image(image_url, temp_filename)

            # Load image to get dimensions safely
            with Image.open(temp_filename) as img:
                img_width, img_height = img.size

            # Run YOLO prediction
            results = self.model(temp_filename)
            task_predictions = []

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                label = CLASS_NAMES.get(class_id, "unknown")

                # Normalize coordinates to percentages
                x = float(x1) / img_width * 100
                y = float(y1) / img_height * 100
                width = float(x2 - x1) / img_width * 100
                height = float(y2 - y1) / img_height * 100

                task_predictions.append(
                    {
                        "result": [
                            {
                                "from_name": "label",
                                "to_name": "image",
                                "type": "rectanglelabels",
                                "value": {"x": x, "y": y, "width": width, "height": height, "rectanglelabels": [label]},
                            }
                        ],
                        "score": conf,
                    }
                )
                print(f"{label=}")

            predictions.extend(task_predictions)

            # Clean up temporary file
            try:
                os.remove(temp_filename)
            except Exception as e:
                print(f"Warning: Failed to delete temp file {temp_filename}: {e}")

        return predictions
