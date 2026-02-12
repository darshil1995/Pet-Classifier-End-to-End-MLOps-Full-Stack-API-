import os
import shutil
from configs import config


def organize_images():
    # Path to the extracted 'train' folder containing 25,000 images
    train_dir = config.EXTRACTED_PATH

    # New subdirectories for classes
    categories = ['cat', 'dog']

    for category in categories:
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)

    print("Organizing images into subfolders...")

    # Loop through all files in the train directory
    for filename in os.listdir(train_dir):
        # Ignore the new folders we just created
        if filename in categories:
            continue

        file_path = os.path.join(train_dir, filename)

        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            if filename.startswith('cat'):
                shutil.move(file_path, os.path.join(train_dir, 'cat', filename))
            elif filename.startswith('dog'):
                shutil.move(file_path, os.path.join(train_dir, 'dog', filename))

    print("Done! Data is now ready for training.")


if __name__ == "__main__":
    organize_images()