import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# === Dice Loss and IOU metric for U-Net ===
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))

def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-6)

# === U-Net Architecture ===
def unet(sz=(256, 256, 3)):
    x = Input(sz)
    inputs = x

    f = 8
    layers = []

    for i in range(6):
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        layers.append(x)
        x = MaxPooling2D()(x)
        f *= 2

    ff2 = 64
    j = len(layers) - 1

    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
    x = Concatenate(axis=3)([x, layers[j]])
    j -= 1

    for i in range(5):
        ff2 //= 2
        f //= 2
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=3)([x, layers[j]])
        j -= 1

    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(), loss=dice_loss, metrics=[mean_iou])
    return model

# === Load both models ===
@st.cache_resource
def load_models():
    classifier = load_model("best_model.keras")
    segmenter = unet()
    segmenter.load_weights("unet.weights.h5")
    return classifier, segmenter

classifier_model, unet_model = load_models()

# Class labels and summaries
class_labels = [
    "AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"
]

class_summaries = {
    "AKIEC": "Precancerous lesion. May develop into skin cancer. Please consult a dermatologist.",
    "BCC": "Basal Cell Carcinoma – a common skin cancer. Slow-growing but should be professionally treated.",
    "BKL": "Benign Keratosis – non-cancerous, but monitor for changes.",
    "DF": "Dermatofibroma – a benign skin lump. Usually harmless.",
    "MEL": "Melanoma – serious and aggressive skin cancer. Seek immediate medical advice.",
    "NV": "Melanocytic Nevus (mole) – typically benign, but monitor for changes in color, size, or shape.",
    "VASC": "Vascular lesion – often benign (e.g., angiomas), though may require treatment if problematic."
}

st.title("🧠 Skin Lesion Diagnosis")

uploaded_file = st.file_uploader("Upload a skin image (JPEG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # === Classification ===
    img_cls = image.resize((128, 128))
    img_cls_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img_cls))
    img_cls_batch = np.expand_dims(img_cls_array, axis=0)

    preds = classifier_model.predict(img_cls_batch)
    pred_class_index = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))
    diagnosis = class_labels[pred_class_index]

    st.markdown(f"### 🧾 **Predicted Class:** {diagnosis}")
    st.markdown(f"**Confidence:** {confidence:.2%}")

    st.markdown("### 🩺 Diagnostic Summary:")
    st.info(class_summaries.get(diagnosis, "No summary available for this class."))

    # === Segmentation ===
    img_seg = image.resize((256, 256))
    img_seg_array = np.array(img_seg) / 255.0
    img_seg_batch = np.expand_dims(img_seg_array, axis=0)

    pred_mask = unet_model.predict(img_seg_batch)[0, :, :, 0]
    mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_binary, image.size, interpolation=cv2.INTER_NEAREST)

    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_cv, contours, -1, (0, 255, 0), thickness=2)

    output_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    st.image(output_image, caption="Segmentation Output (Green Borders)", use_column_width=True)