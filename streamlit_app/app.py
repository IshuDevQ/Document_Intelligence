import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image


API_URL = os.environ.get("API_URL", "http://localhost:8000")


st.set_page_config(
    page_title="Document Intelligence System",
    layout="wide",
)

st.title("Document Intelligence System")
st.write("Image tampering detection using OpenCV preprocessing, ELA, OCR, and Random Forest.")

with st.sidebar:
    st.header("Backend Settings")
    st.write("API URL:")
    st.code(API_URL)

    if st.button("Check API Health"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=10)

            if response.status_code == 200:
                st.success("API is reachable.")
                st.json(response.json())
            else:
                st.error(response.text)

        except Exception as error:
            st.error(f"Could not connect to API: {error}")


tab_verify, tab_history, tab_stats, tab_model = st.tabs(
    [
        "Verify Document",
        "History",
        "Statistics",
        "Model Performance",
    ]
)


with tab_verify:
    st.subheader("Upload Document Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Prediction")

            if st.button("Verify Document", type="primary"):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                try:
                    with st.spinner("Analyzing document..."):
                        response = requests.post(
                            f"{API_URL}/verify",
                            files=files,
                            timeout=120,
                        )

                    if response.status_code != 200:
                        st.error(response.text)

                    else:
                        result = response.json()

                        if result["label"] == "tampered":
                            st.error(result["verdict"])
                        elif result["is_uncertain"]:
                            st.warning(result["verdict"])
                        else:
                            st.success(result["verdict"])

                        metric_col1, metric_col2, metric_col3 = st.columns(3)

                        metric_col1.metric(
                            "Prediction",
                            result["label"],
                        )

                        metric_col2.metric(
                            "Confidence",
                            f"{result['confidence']:.2%}",
                        )

                        metric_col3.metric(
                            "Processing Time",
                            f"{result['processing_time_ms']} ms",
                        )

                        st.subheader("Probabilities")

                        st.write(
                            "Authentic probability:",
                            result["authentic_prob"],
                        )

                        st.progress(result["authentic_prob"])

                        st.write(
                            "Tampered probability:",
                            result["tampered_prob"],
                        )

                        st.progress(result["tampered_prob"])

                        st.subheader("ELA Map")

                        ela_path = Path(result["ela_image_path"])

                        if ela_path.exists():
                            ela_image = Image.open(ela_path)
                            st.image(
                                ela_image,
                                caption="Error Level Analysis Map",
                                use_container_width=True,
                            )
                        else:
                            st.warning("ELA image was generated but could not be found locally.")

                        st.subheader("ELA Features")

                        st.write("ELA mean:", result["ela_mean"])
                        st.write("ELA region standard deviation:", result["ela_region_std"])

                        st.subheader("OCR Result")

                        st.write("OCR confidence:", result["ocr_confidence"])

                        st.text_area(
                            "Extracted Text",
                            result["ocr_text"],
                            height=180,
                        )

                except Exception as error:
                    st.error(f"Request failed: {error}")


with tab_history:
    st.subheader("Verification History")

    limit = st.slider(
        "Number of records",
        min_value=1,
        max_value=100,
        value=20,
    )

    if st.button("Load History"):
        try:
            response = requests.get(
                f"{API_URL}/history",
                params={"limit": limit},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                results = data["results"]

                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No history found yet.")

            else:
                st.error(response.text)

        except Exception as error:
            st.error(f"Could not load history: {error}")


with tab_stats:
    st.subheader("System Statistics")

    if st.button("Load Statistics"):
        try:
            response = requests.get(f"{API_URL}/stats", timeout=30)

            if response.status_code == 200:
                data = response.json()

                col1, col2, col3 = st.columns(3)

                col1.metric("Total", data["total"])
                col2.metric("Authentic", data["total_authentic"])
                col3.metric("Tampered", data["total_tampered"])

                col4, col5, col6 = st.columns(3)

                col4.metric("Uncertain", data["total_uncertain"])
                col5.metric("Avg Confidence", data["avg_confidence"])
                col6.metric("Avg Processing ms", data["avg_processing_ms"])

                st.json(data)

            else:
                st.error(response.text)

        except Exception as error:
            st.error(f"Could not load statistics: {error}")


with tab_model:
    st.subheader("Model Performance")

    if st.button("Load Model Performance"):
        try:
            response = requests.get(f"{API_URL}/model-performance", timeout=30)

            if response.status_code == 200:
                data = response.json()

                if data["accuracy"] == 0:
                    st.warning("No model performance found. Train the model first.")
                else:
                    col1, col2, col3 = st.columns(3)

                    col1.metric("Accuracy", f"{data['accuracy']:.2%}")
                    col2.metric("Features", data["n_features"])
                    col3.metric("Trees", data["n_trees"])

                    col4, col5 = st.columns(2)

                    col4.metric("Train Samples", data["train_samples"])
                    col5.metric("Test Samples", data["test_samples"])

                    st.subheader("Class-wise Performance")

                    performance_df = pd.DataFrame(
                        [
                            {
                                "class": "authentic",
                                "precision": data["authentic_precision"],
                                "recall": data["authentic_recall"],
                                "f1_score": data["authentic_f1"],
                            },
                            {
                                "class": "tampered",
                                "precision": data["tampered_precision"],
                                "recall": data["tampered_recall"],
                                "f1_score": data["tampered_f1"],
                            },
                        ]
                    )

                    st.dataframe(performance_df, use_container_width=True)

                    st.subheader("Raw Model Performance")
                    st.json(data)

            else:
                st.error(response.text)

        except Exception as error:
            st.error(f"Could not load model performance: {error}")