import os
import tensorflow as tf
from config.config import Config

# set TF environment before loading
os.environ["TF_CPP_MIN_LOG_LEVEL"] = Config.TF_LOG_LEVEL
os.environ["TF_ENABLE_ONEDNN_OPTS"] = Config.DISABLE_ONEDNN

_model = None


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(Config.MODEL_PATH)
    return _model
