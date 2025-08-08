import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from utils.model_utils import get_model
from config.config import Config
from utils.preprocess_pil import preprocess_pil

# Configure environment variables for TensorFlow verbosity and oneDNN
os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL", os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")
)
os.environ.setdefault(
    "TF_ENABLE_ONEDNN_OPTS", os.environ.get("TF_ENABLE_ONEDNN_OPTS", "0")
)


# Create Flask app

app = Flask(__name__)
CORS(app)


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "good"}), 200


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    b64 = data.get("image") or data.get("b64")
    if not b64:
        return jsonify({"error": "No image data provided"}), 400

    try:
        if b64.startswith("data:image"):
            b64 = b64.split(",", 1)[1]

        img = Image.open(BytesIO(base64.b64decode(b64)))
        x = preprocess_pil(img)

        model = get_model()
        if not model:
            return jsonify({"error": "Model not loaded"}), 500
        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs)) * 100.0

        return jsonify(
            {
                "prediction": pred,
                "confidence": round(conf, 2),
                "debug": {
                    "input_shape": list(x.shape),
                    "invert": Config.INVERT,
                    "norm": Config.NORM,
                },
            }
        )
    except Exception as e:
        return jsonify({"error": "Failed to process image", "details": str(e)}), 500


# For local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False)
