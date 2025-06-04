# PEK Automotive Test

This program allows the user to upload multiple images and searches for apples using a YOLO model. For every image, the application shows the original image, the annotated one, and a table where each row corresponds to a detected apple. The table includes information about the coordinates of the center of the bounding box in px, the accuracy of the detection, and the area of the bounding box in px².

With only RGB images and no depth information, the only way to estimate the distance of the apples from the camera is by using the area of the detection. The larger the area, the closer the apple is assumed to be to the camera.
The list of apples is sorted by decreasing area, so the closest apple is always at the top of the list.

## Installation

Make sure you have Python installed, then install the required packages using:

```bash
pip install -r requirements.txt
```

Start the application with:

```bash
streamlit run main.py
```

You can then access the application in your web browser at http://localhost:8501.

## Demo Video

[Watch the demo video](https://drive.google.com/file/d/1YHRO1SkyuEjG-2olhQJSVyck78xrYE8-/view?usp=sharing)
