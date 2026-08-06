"""Streamlit app: run any YOLO26 model (or Meta's SAM3) on an uploaded image
or an image URL.

Run from the project root:
    ./venv/bin/streamlit run cnn/yolo/app.py

Pick a task (detection, segmentation, depth, classification, pose, ...) and a
model size in the sidebar - the weights are downloaded automatically by
Ultralytics on first use and cached in cnn/yolo/weights/, so each model
downloads only once.
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics.models.sam import SAM3SemanticPredictor

# Where downloaded .pt weight files are stored, next to this script.
WEIGHTS_DIR = Path(__file__).parent / 'weights'
WEIGHTS_DIR.mkdir(exist_ok=True)

# Task -> filename suffix used by Ultralytics.
# e.g. Detection + size 'n' -> yolo26n.pt, Pose + 'x' -> yolo26x-pose.pt
TASKS = {
    'Detection': '',
    'Instance Segmentation': '-seg',
    'Semantic Segmentation': '-sem',
    'Depth Estimation': '-depth',
    'Classification': '-cls',
    'Pose Estimation': '-pose',
    'Oriented Detection (OBB)': '-obb',
}

# SAM3 is a separate model family (Meta's Segment Anything 3, run through
# Ultralytics). Instead of fixed classes it segments whatever CONCEPT you
# describe with free text, e.g. "red car" or "person wearing a hat".
SAM3_TASK = 'SAM3 - Segment by Text (Segment Anything 3)'
SAM3_WEIGHTS = WEIGHTS_DIR / 'sam3.pt'

# Model sizes: n(ano) is fastest/least accurate, x is slowest/most accurate.
SIZES = {
    'n (nano - fastest)': 'n',
    's (small)': 's',
    'm (medium)': 'm',
    'l (large)': 'l',
    'x (xlarge - most accurate)': 'x',
}


@st.cache_resource   # one cache entry PER model name - switching is instant after first load
def load_model(model_name: str) -> YOLO:
    # Passing a path inside WEIGHTS_DIR makes Ultralytics download the
    # weights there (instead of littering the project root).
    return YOLO(str(WEIGHTS_DIR / model_name))


@st.cache_resource
def load_sam3() -> SAM3SemanticPredictor:
    # SAM3 text ("concept") segmentation is not exposed through the plain
    # YOLO/SAM classes - it needs the semantic predictor directly.
    # Loaded once and cached (the checkpoint is ~3 GB); the confidence
    # threshold is updated per-run in the inference code below.
    return SAM3SemanticPredictor(overrides=dict(
        model=str(SAM3_WEIGHTS), task='segment', mode='predict',
        imgsz=1024, conf=0.25, save=False, verbose=False))


@st.cache_data(show_spinner='Downloading image...')
def fetch_image(url: str) -> bytes:
    response = requests.get(url, timeout=15,
                            headers={'User-Agent': 'Mozilla/5.0 (yolo26-demo)'})
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------------------------
# Sidebar: model choice
# ---------------------------------------------------------------------------
st.set_page_config(page_title='YOLO26 Playground', page_icon='🎯', layout='wide')
st.title('YOLO26 Playground')
st.write('Pick a task and model size in the sidebar, then upload an image '
         'or paste an image URL.')

with st.sidebar:
    st.header('Model')
    task = st.selectbox('Task', list(TASKS.keys()) + [SAM3_TASK])
    # Confidence threshold: predictions below this score are discarded.
    # (Not used by classification/depth, which always produce one output.)
    conf = st.slider('Confidence threshold', 0.05, 0.95, 0.25, 0.05)

    if task == SAM3_TASK:
        # SAM3 comes in one size only, and its weights are license-gated by
        # Meta, so they cannot be auto-downloaded like the YOLO26 ones.
        sam3_text = st.text_input(
            'What to segment (free text)', placeholder='e.g. red car, person, dog',
            help='Describe the concept(s) to find. Separate several with commas.')
        model_name = 'sam3.pt'
        st.caption('Model file: `sam3.pt` - must be downloaded manually '
                   '(see instructions on the main page if missing).')
    else:
        size_label = st.selectbox('Size', list(SIZES.keys()))
        model_name = f'yolo26{SIZES[size_label]}{TASKS[task]}.pt'
        st.caption(f'Model file: `{model_name}` - downloads automatically on '
                   'first use (may take a moment for large sizes).')

# ---------------------------------------------------------------------------
# Getting an image in: upload or URL, same as the CIFAR-10 app.
# ---------------------------------------------------------------------------
tab_upload, tab_url = st.tabs(['Upload a file', 'Paste an image URL'])

image = None
with tab_upload:
    uploaded = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg', 'bmp', 'webp'])
    if uploaded is not None:
        image = Image.open(uploaded)

with tab_url:
    url = st.text_input('Image URL', placeholder='https://example.com/street.jpg')
    if url:
        try:
            image = Image.open(BytesIO(fetch_image(url)))
        except requests.RequestException as e:
            st.error(f'Could not download the image: {e}')
        except Exception:
            st.error('The URL did not return a valid image file. Make sure it '
                     'points directly to an image (e.g. ends in .jpg or .png).')

# ---------------------------------------------------------------------------
# Inference + display
# ---------------------------------------------------------------------------
# SAM3 needs its weights in place and a text prompt before it can run.
if task == SAM3_TASK and not SAM3_WEIGHTS.exists():
    st.error('SAM3 weights not found.')
    st.markdown(f'''
Meta gates the SAM3 weights behind a license, so they cannot be downloaded
automatically. One-time setup:

1. Open [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
   and accept the license (free account required).
2. Download **`sam3.pt`** (~3 GB).
3. Put it at: `{SAM3_WEIGHTS}`

Then reload this page.''')
    st.stop()

if image is not None:
    image = image.convert('RGB')   # drop alpha channel / handle grayscale

    if task == SAM3_TASK:
        if not sam3_text.strip():
            st.info('Type what to segment in the sidebar (e.g. "dog") '
                    'to run SAM3.')
            st.stop()
        # Comma-separated input -> list of concepts. Each concept becomes
        # its own "class" in the results, in the same order.
        concepts = [t.strip() for t in sam3_text.split(',') if t.strip()]
        with st.spinner('Running SAM3 (large model - slow on CPU)...'):
            predictor = load_sam3()
            predictor.args.conf = conf   # apply the sidebar slider
            result = predictor(source=np.asarray(image), text=concepts)[0]
        # Map class indices back to the typed concepts so the plot and the
        # table show your words instead of bare numbers.
        result.names = {i: c for i, c in enumerate(concepts)}
    else:
        with st.spinner(f'Loading {model_name} and running inference...'):
            model = load_model(model_name)
            # YOLO handles all preprocessing (resize, normalize) internally.
            result = model(image, conf=conf, verbose=False)[0]

    col_in, col_out = st.columns(2)
    with col_in:
        st.subheader('Input')
        st.image(image, use_container_width=True)

    with col_out:
        st.subheader('Result')
        # result.plot() draws the predictions (boxes, masks, keypoints,
        # depth map... depending on the task) and returns a BGR numpy
        # array - [..., ::-1] flips the channel order to RGB for display.
        st.image(result.plot()[..., ::-1], use_container_width=True)

    # ----- task-specific details below the images -----
    if task == 'Classification' and result.probs is not None:
        # Top-5 most likely classes out of ImageNet's 1000.
        st.subheader('Top-5 predictions')
        top5 = result.probs.top5
        top5conf = result.probs.top5conf.tolist()
        st.table({'class': [result.names[i] for i in top5],
                  'confidence': [f'{c:.1%}' for c in top5conf]})

    elif result.boxes is not None and len(result.boxes) > 0:
        # Detection / instance segmentation / pose: list every found object.
        st.subheader(f'Detected objects ({len(result.boxes)})')
        classes = result.boxes.cls.tolist()
        confs = result.boxes.conf.tolist()
        st.table({'object': [result.names[int(c)] for c in classes],
                  'confidence': [f'{c:.1%}' for c in confs]})

    elif result.obb is not None and len(result.obb) > 0:
        # Oriented bounding boxes (aerial imagery classes: plane, ship...).
        st.subheader(f'Detected objects ({len(result.obb)})')
        classes = result.obb.cls.tolist()
        confs = result.obb.conf.tolist()
        st.table({'object': [result.names[int(c)] for c in classes],
                  'confidence': [f'{c:.1%}' for c in confs]})

    elif task in ('Detection', 'Instance Segmentation', 'Pose Estimation',
                  'Oriented Detection (OBB)', SAM3_TASK):
        st.info('No objects found above the confidence threshold - try '
                'lowering it in the sidebar.')

    # Inference timing, e.g. "45.3 ms" - useful for comparing model sizes.
    speed = sum(v for v in result.speed.values() if v)
    st.caption(f'Inference time: {speed:.0f} ms ({model_name}, CPU)')
