import os
import zipfile
from configs import config
from dotenv import load_dotenv
from src.logger import get_logger

# Note: NO Kaggle import at the top!
logger = get_logger(__name__)


def download_and_extract():
    # 1. Load the .env file
    # This looks for the .env file in the root directory where you run main.py
    load_dotenv()

    # 2. Extract and Validate values
    kaggle_user = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_API_TOKEN')

    if not kaggle_user or not kaggle_key:
        logger.error("KAGGLE_USERNAME or KAGGLE_KEY not found in .env file!")
        return

    # 3. Set them into the environment BEFORE importing Kaggle
    os.environ['KAGGLE_USERNAME'] = kaggle_user
    os.environ['KAGGLE_API_TOKEN'] = kaggle_key

    try:
        # 4. Import Kaggle ONLY inside the function
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        logger.info("Kaggle authentication successful.")

        if not os.path.exists(config.RAW_DATA_PATH):
            os.makedirs(config.RAW_DATA_PATH)

        zip_path = os.path.join(config.RAW_DATA_PATH, 'dogs-vs-cats.zip')

        if not os.path.exists(zip_path):
            logger.info("Downloading dataset...")
            # Note: Ensure you have accepted the competition rules on Kaggle.com
            api.competition_download_files('dogs-vs-cats', path=config.RAW_DATA_PATH)

        # Extraction logic
        logger.info("Extracting main zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(config.RAW_DATA_PATH)

        train_zip = os.path.join(config.RAW_DATA_PATH, 'train.zip')
        if os.path.exists(train_zip):
            logger.info("Extracting training images...")
            with zipfile.ZipFile(train_zip, 'r') as zip_ref:
                zip_ref.extractall(config.DATA_DIR)

        logger.info(f"Pipeline ready. Data in {config.EXTRACTED_PATH}")

    except Exception as e:
        logger.error(f"Kaggle API Error: {e}")
        raise


if __name__ == "__main__":
    download_and_extract()