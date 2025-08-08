import os


class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "mlp_mnist_model.h5")
    TF_LOG_LEVEL = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")
    DISABLE_ONEDNN = os.environ.get("TF_ENABLE_ONEDNN_OPTS", "0")
    # ====== CONFIG based on your training ======
    FLAT_INPUT = True  # (1, 784)
    NORM = "0_1"  # divide by 255.0
    INVERT = True  # canvas black-on-white -> invert -> white-on-black (MNIST style)
    # ===========================================
