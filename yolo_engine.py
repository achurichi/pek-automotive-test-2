from ultralytics import YOLO


class YoloEngine:
    """
    Wrapper class for loading a YOLO model and running batch predictions.
    """

    def __init__(self, model_path="yolo11m.pt"):
        """
        Initialize the YoloEngine by loading the YOLO model.

        Args:
            model_path (str): Path to the YOLO model weights file.
        """
        self.model = YOLO(model_path)

    def predict_batch(self, images):
        """
        Run YOLO batch prediction on a list of images.

        Args:
            images (list): List of images (numpy arrays) to predict on.

        Returns:
            list: List of YOLO prediction results for each image.
        """
        if not images:
            return []
        return self.model(images)

    def get_label(self, class_idx):
        """
        Get the class label string for a given class index.

        Args:
            class_idx (int): Index of the class.

        Returns:
            str: Class label name.
        """
        return self.model.names[class_idx]
