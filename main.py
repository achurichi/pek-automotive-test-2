from apple_detector import AppleDetector
from yolo_engine import YoloEngine
import cv2
import pandas as pd
import streamlit as st

MAX_NUM_FILES = 20
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@st.cache_resource
def get_yolo_engine():
    """
    Load and cache the YOLO engine instance for efficient reuse.

    Returns:
        YoloEngine: An instance of the YoloEngine class with the loaded model.
    """
    return YoloEngine()


def display_results(images_info):
    """
    Display original and annotated images, along with detection tables, in Streamlit.

    Args:
        images_info (list): List of dictionaries containing image data, annotations, and apple detection results.
    """
    for idx, info in enumerate(images_info):
        if idx != 0:
            st.markdown("---")

        st.subheader(info["name"])
        col1, col2 = st.columns(2)
        if info["image"] is not None:
            with col1:
                st.image(
                    cv2.cvtColor(info["image"], cv2.COLOR_BGR2RGB),
                    caption="Original",
                    use_container_width=True,
                )
        if info["annotated"] is not None:
            with col2:
                st.image(
                    cv2.cvtColor(info["annotated"], cv2.COLOR_BGR2RGB),
                    caption="Annotated",
                    use_container_width=True,
                )

        if info["apple_data"]:
            sorted_apple_data = sorted(
                info["apple_data"], key=lambda x: x["Area (px²)"], reverse=True
            )
            st.write("Apple Data")
            st.dataframe(pd.DataFrame(sorted_apple_data), hide_index=True)
        else:
            st.info("No apples detected.")


if __name__ == "__main__":
    st.set_page_config(page_title="Pek Automotive Test", layout="wide")

    uploads = st.sidebar.file_uploader(
        "Upload images", accept_multiple_files=True, type=["png", "jpg", "jpeg"]
    )

    if uploads:
        if len(uploads) > MAX_NUM_FILES:
            st.error(f"You can upload a maximum of {MAX_NUM_FILES} files at a time.")
            st.stop()

        for upload in uploads:
            if upload.size > MAX_FILE_SIZE:
                st.error(
                    f"The uploaded file {upload.name} is too large. Please upload an image smaller than {MAX_FILE_SIZE/1024/1024:.1f}MB."
                )

        valid_uploads = [u for u in uploads if u.size <= MAX_FILE_SIZE]
        if not valid_uploads:
            st.stop()

        yolo_engine = get_yolo_engine()
        detector = AppleDetector(yolo_engine)

        with st.spinner("Processing images...", show_time=True):
            images_info = detector.parse_uploaded_images(valid_uploads)

            # Handle images that couldn't be loaded
            failed = [info["name"] for info in images_info if info["image"] is None]
            for name in failed:
                st.error(f"Failed to load image: {name}")

            images_info = detector.predict_batch(images_info)
            images_info = detector.prepare_apple_results(images_info)
        display_results(images_info)
