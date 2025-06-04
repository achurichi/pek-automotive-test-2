import cv2
import numpy as np


class AppleDetector:
    def __init__(self, yolo_engine):
        """
        Initialize the AppleDetector with a YOLO engine instance.

        Args:
            yolo_engine (YoloEngine): An instance of the YoloEngine class for inference.
        """
        self.yolo_engine = yolo_engine

    def parse_uploaded_images(self, uploads):
        """
        Parse uploaded files into image info dictionaries, decoding images with OpenCV.

        Args:
            uploads (list): List of uploaded file-like objects.

        Returns:
            list: List of dictionaries with image data and metadata.
        """
        images_info = []

        for upload in uploads:
            try:
                image_bytes = upload.getvalue()
                np_arr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"ERROR: Failed to decode image {upload.name}: {e}")
                image = None

            images_info.append(
                {
                    "name": upload.name,
                    "image": image,
                    "predictions": None,
                    "annotated": None,
                    "apple_data": None,
                }
            )

        return images_info

    def predict_batch(self, images_info):
        """
        Run YOLO batch prediction on images and store results in images_info.

        Args:
            images_info (list): List of dictionaries with image data.

        Returns:
            list: Updated images_info with YOLO predictions.
        """
        valid_infos = [info for info in images_info if info["image"] is not None]
        images = [info["image"] for info in valid_infos]

        try:
            results = self.yolo_engine.predict_batch(images)
        except Exception as e:
            print(f"ERROR: YOLO batch prediction failed: {e}")
            results = [None] * len(valid_infos)

        for info, result in zip(valid_infos, results):
            info["predictions"] = result

        return images_info

    def prepare_apple_results(self, images_info):
        """
        Annotate images with bounding boxes and extract apple detection data.

        Args:
            images_info (list): List of dictionaries with image and prediction data.

        Returns:
            list: Updated images_info with annotated images and apple data.
        """
        for info in images_info:
            image = info["image"]
            result = info["predictions"]
            if image is None or result is None:
                continue

            try:
                annotated = image.copy()
                thickness = max(2, min(image.shape[0], image.shape[1]) // 400)
                font_scale = max(0.5, min(image.shape[0], image.shape[1]) / 1000.0)
                apple_data = []
                apple_count = 1

                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = self.yolo_engine.get_label(cls)

                    # Filter out non-apple detections
                    if label != "apple":
                        continue

                    # Get the confidence and the bbox area
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    area = (x2 - x1) * (y2 - y1)

                    # Draw bbox with number
                    self.draw_numbered_bbox(
                        annotated, x1, y1, x2, y2, apple_count, thickness, font_scale
                    )

                    # Save apple data
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    apple_data.append(
                        {
                            "Number": apple_count,
                            "Position (px)": f"({cx}, {cy})",
                            "Accuracy": f"{conf:.2f}",
                            "Area (px²)": area,
                        }
                    )
                    apple_count += 1

                info["annotated"] = annotated
                info["apple_data"] = apple_data

            except Exception as e:
                print(f"ERROR: Failed to process image {info['name']}: {e}")
                info["annotated"] = None
                info["apple_data"] = []

        return images_info

    @staticmethod
    def draw_numbered_bbox(image, x1, y1, x2, y2, number, thickness, font_scale):
        """
        Draw a red bounding box and a numbered label with a black background on the image.

        Args:
            image (ndarray): The image to annotate (modified in place).
            x1, y1, x2, y2 (int): Bounding box coordinates.
            number (int): The number to display inside the box.
            thickness (int): Thickness of the bounding box and text.
            font_scale (float): Font scale for the label text.
        """
        # Draw the bounding box in red
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness)

        label_text = str(number)

        # Calculate text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Default: place above the box
        rect_x1 = x1 + 2
        rect_x2 = rect_x1 + text_w + 6
        rect_y2 = y1 - 4
        rect_y1 = rect_y2 - (text_h + baseline + 4)
        # If the rectangle would go out of the image (top), place below the box
        if rect_y1 < 0:
            rect_y1 = y2 + 4
            rect_y2 = rect_y1 + text_h + baseline + 4

        # Draw black rectangle as background for the label
        cv2.rectangle(
            image,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            (0, 0, 0),
            -1,
        )

        # Draw the label text in red on top of the black background
        cv2.putText(
            image,
            label_text,
            (rect_x1 + 3, rect_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            thickness,
        )
