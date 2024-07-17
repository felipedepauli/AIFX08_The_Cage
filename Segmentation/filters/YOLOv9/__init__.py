import cv2
import numpy as np
import os
from ultralytics import YOLO
from filters.Filter import Filter

class YOLOv9Filter(Filter):
    def __init__(self, config):
        self.config = config
        self.model = YOLO(self.config.get("model_path", "yolov9-seg.pt"))

    def filter(self, image):
        results = self.model(image)
        processed_image = image.copy()
        self.intermediate_images = {
            "Original Image": image,
            "Processed Image": processed_image
        }

        for result in results:
            for det in result.masks.data:  # Assume que o modelo YOLOv9 retorna máscaras
                mask = det.cpu().numpy()
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask = (mask > 0.5).astype(np.uint8)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(processed_image, contours, -1, (0, 255, 0), 2)

        self.intermediate_images["Processed Image"] = processed_image

        return processed_image

    def save_intermediate_images(self, output_dir):
        for name, img in self.intermediate_images.items():
            path = os.path.join(output_dir, f"{name.replace(' ', '_')}.jpg")
            cv2.imwrite(path, img)
