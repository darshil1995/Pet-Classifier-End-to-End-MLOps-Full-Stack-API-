import os
from dotenv import load_dotenv
from configs import config

"""This ensures your environment is set up correctly before the long training process starts."""

def test_env_variables():
    """Check if Kaggle credentials are loaded."""
    load_dotenv()
    assert os.getenv('KAGGLE_USERNAME') is not None, "KAGGLE_USERNAME not in .env"
    assert os.getenv('KAGGLE_API_TOKEN') is not None, "KAGGLE_API_TOKEN not in .env"

def test_path_validity():
    """Ensure paths in config are strings and not empty."""
    assert isinstance(config.MODEL_PATH, str)
    assert len(config.MODEL_PATH) > 0
    assert config.MODEL_NAME.endswith('.keras')

def test_directory_creation():
    """Ensure required directories exist or are createable."""
    dirs = [config.MODEL_SAVE_DIR, 'logs', config.DATA_DIR]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
        assert os.path.exists(d), f"Directory {d} could not be verified."