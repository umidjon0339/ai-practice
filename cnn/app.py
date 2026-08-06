"""Streamlit app: upload an image (or paste a URL) and classify it with the
trained CIFAR-10 CNN.

Run from the project root:
    ./venv/bin/streamlit run cnn/app.py

How Streamlit works: it re-runs this WHOLE script top-to-bottom on every
user interaction (every click, upload, keystroke). That is why caching
decorators (@st.cache_resource / @st.cache_data) matter - they keep
expensive things (the model, downloads) from being redone on every rerun.
"""

from io import BytesIO      # wraps downloaded bytes so PIL can open them like a file
from pathlib import Path

import numpy as np
import requests             # for downloading images from a URL
import streamlit as st      # every st.* call renders a UI element
import tensorflow as tf
from PIL import Image       # image loading/resizing

# Build the model path relative to THIS file's folder, so the app works no
# matter which directory you launch streamlit from.
MODEL_PATH = Path(__file__).parent / 'cifar10_cnn.keras'

# Must be in the SAME order as in train_cnn.py - the model only outputs
# class numbers 0..9; this list translates them to words.
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


@st.cache_resource   # load the model from disk ONCE, reuse it on every rerun
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    """Convert an uploaded image to the (1, 32, 32, 3) float tensor the CNN expects.

    This must mirror the training preprocessing exactly:
    - convert('RGB'): handles PNGs with transparency and grayscale images
    - resize to 32x32: the only size the model was trained on
    - / 255.0: same normalization as training (0..255 -> 0..1)
    - [np.newaxis]: adds a batch dimension, (32,32,3) -> (1,32,32,3),
      because the model always expects a BATCH of images, even a batch of 1.
    """
    image = image.convert('RGB').resize((32, 32))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array[np.newaxis, ...]


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.set_page_config(page_title='CIFAR-10 CNN', page_icon='🖼️')
st.title('CIFAR-10 Image Classifier')
st.write('Upload an image and the CNN will predict which of the 10 CIFAR-10 '
         'classes it belongs to: ' + ', '.join(class_names) + '.')

# Guard: without this, a missing model file would crash load_model() with a
# confusing traceback. Show a helpful message and stop the script instead.
if not MODEL_PATH.exists():
    st.error(f'Model not found at {MODEL_PATH}. Train it first with '
             '`python cnn/train_cnn.py`.')
    st.stop()

model = load_model()


@st.cache_data(show_spinner='Downloading image...')   # remember result per URL
def fetch_image(url: str) -> bytes:
    """Download raw bytes from a URL.

    - timeout=15: don't hang forever on a dead server
    - User-Agent header: look like a browser; some sites reject bare scripts
    - raise_for_status(): turn HTTP errors (404 etc.) into Python exceptions
      that the caller can catch and display nicely.
    """
    response = requests.get(url, timeout=15,
                            headers={'User-Agent': 'Mozilla/5.0 (cifar10-demo)'})
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------------------------
# Getting an image in - two tabs, one result.
# Both tabs do the same job: put a PIL Image into the `image` variable.
# ---------------------------------------------------------------------------
tab_upload, tab_url = st.tabs(['Upload a file', 'Paste an image URL'])

image = None
with tab_upload:
    uploaded = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg', 'bmp', 'webp'])
    if uploaded is not None:
        image = Image.open(uploaded)

with tab_url:
    url = st.text_input('Image URL', placeholder='https://example.com/cat.jpg')
    if url:
        try:
            image = Image.open(BytesIO(fetch_image(url)))
        except requests.RequestException as e:
            # network problems: bad URL, 404, timeout, blocked by the site...
            st.error(f'Could not download the image: {e}')
        except Exception:
            # downloaded fine, but the content is not an image (e.g. HTML page)
            st.error('The URL did not return a valid image file. Make sure it '
                     'points directly to an image (e.g. ends in .jpg or .png).')

# ---------------------------------------------------------------------------
# Prediction + display - runs only once we have an image, from either tab.
# ---------------------------------------------------------------------------
if image is not None:
    col_img, col_pred = st.columns(2)   # show image and result side by side
    with col_img:
        st.image(image, caption='Uploaded image', use_container_width=True)

    # The model returns 10 raw scores (logits). Softmax converts them into
    # probabilities that sum to 1. We must apply it HERE because training
    # used from_logits=True - the model itself never applies softmax.
    logits = model.predict(preprocess(image), verbose=0)
    probs = tf.nn.softmax(logits[0]).numpy()
    best = int(np.argmax(probs))        # index of the highest probability

    with col_pred:
        st.subheader('Prediction')
        st.metric(label='Class', value=class_names[best],
                  delta=f'{probs[best]:.1%} confidence')

    # Bar chart of all 10 probabilities, sorted highest first. Useful for
    # spotting when the model is torn between two classes (45% cat / 40% dog).
    st.subheader('Class probabilities')
    order = np.argsort(probs)[::-1]
    st.bar_chart({'probability': {class_names[i]: float(probs[i]) for i in order}})

    st.caption('Note: the model was trained on 32x32 CIFAR-10 images, so your '
               'upload is downscaled to 32x32 before prediction. Images unlike '
               'the 10 training classes will still be forced into one of them.')
