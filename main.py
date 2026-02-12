import sys
import pytest
from src.data_loader import download_and_extract
from src.organize_data import organize_images
from src.train import train
from src.logger import get_logger

logger = get_logger(__name__)

"""
How to use it:
# Run from the root directory
python -m src.predict "C:/Users/Darshil/Pictures/my_dog.jpg"
"""


def run_validation(stage_name, test_files):
    """
    Helper function to run specific test files and handle exit codes.
    """
    logger.info(f">>> Running {stage_name} Validation...")

    # Run pytest on the list of files provided
    # -x: stop on first failure, -q: quiet (less clutter in console)
    exit_code = pytest.main(["-x", "-q"] + test_files)

    if exit_code == 0:
        logger.info(f">>> {stage_name} Passed!")
        return True
    else:
        logger.error(f">>> {stage_name} Failed! Please fix issues before proceeding.")
        return False


def main():
    # --- STAGE 1: Infrastructure ---
    # Tests architecture and .env (No images or saved models needed)
    if not run_validation("Infrastructure", ["tests/test_config.py"]):
        sys.exit(1)

    try:
        # --- STEP 2: Setup ---
        download_and_extract()
        organize_images()

        # --- STAGE 3: Data Integrity ---
        # Tests if images were moved correctly (Needs images, but not a saved model)
        if not run_validation("Data Integrity", ["tests/test_data.py"]):
            sys.exit(1)

        # --- STEP 4: Execution ---
        train()  # This creates the .keras file

        # --- STAGE 5: Final Artifact Check ---
        # NOW we test if the model can be loaded and used for prediction
        logger.info(">>> Stage 3: Verifying Saved Model...")
        if not run_validation("Verifying Saved Model", ["tests/test_model.py"]):
            sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline encountered a critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


