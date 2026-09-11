"""
Model Insights page.

Streamlit auto-detects any file inside a `pages/` folder next to app.py and
adds it as a page in the sidebar — no changes to app.py are needed.

This page does NOT retrain or run the model. It just displays static assets
(PNG charts + a JSON summary) that were exported once from the training
notebook in Colab. See colab_export_insights.py for how those files are made.
"""

from pathlib import Path
import json

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

st.set_page_config(page_title="Model Insights", page_icon="📊")
st.title("Model Insights")
st.write(
    "A look at the dataset and model behind the digit recognizer — "
    "how much data it trained on, how balanced it was, and where it "
    "still makes mistakes."
)


def show_image(filename: str, caption: str) -> None:
    """Display an asset image if it exists, otherwise show a friendly message."""
    path = ASSETS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"'{filename}' not found in assets/ yet — export it from Colab first.")


# ---------------------------------------------------------------
# Summary numbers
# ---------------------------------------------------------------
summary_path = ASSETS_DIR / "dataset_summary.json"
if summary_path.exists():
    with open(summary_path) as f:
        summary = json.load(f)

    col1, col2, col3 = st.columns(3)
    col1.metric("Training images", f"{summary['train_samples']:,}")
    col2.metric("Test images", f"{summary['test_samples']:,}")
    col3.metric("Test accuracy", f"{summary['final_test_accuracy'] * 100:.2f}%")
else:
    st.info("'dataset_summary.json' not found in assets/ yet — export it from Colab first.")

st.divider()

# ---------------------------------------------------------------
# Dataset visuals
# ---------------------------------------------------------------
st.header("The Dataset")

st.subheader("Sample digits")
show_image("sample_digits.png", "A random sample of the training images.")

st.subheader("Class balance")
st.write("How evenly the digits 0–9 are represented in the training data.")
show_image("class_distribution.png", "Number of training examples per digit.")

st.divider()

# ---------------------------------------------------------------
# Training visuals
# ---------------------------------------------------------------
st.header("Training")
show_image("training_curves.png", "Accuracy and loss over training epochs.")

st.divider()

# ---------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------
st.header("Where the Model Struggles")

st.subheader("Confusion matrix")
st.write("Rows are the true digit, columns are what the model predicted.")
show_image("confusion_matrix.png", "CNN confusion matrix on the test set.")

st.subheader("Misclassified examples")
st.write("Real test images the model got wrong — many are genuinely ambiguous handwriting.")
show_image("misclassified.png", "A sample of misclassified digits.")

st.subheader("Per-class performance")
show_image("per_class_metrics.png", "Precision, recall, and F1 score for each digit.")