import click
import os
from PIL import Image
import matplotlib.pyplot as plt

@click.command()
@click.option('--dataset_folder', type=str, default='dataset', help='dataset folder')
def main(dataset_folder):
    # Visualize a few photos
    dolphin_folder = os.path.join(dataset_folder, "1_dolphin")

    # Check if the folder exists
    if not os.path.exists(dolphin_folder):
        print(f"Error: The folder {dolphin_folder} does not exist.")
        return

    # Get all file names in the folder
    image_files = [f for f in os.listdir(dolphin_folder) if os.path.isfile(os.path.join(dolphin_folder, f))]

    # Select a few samples (e.g., the first 5 images)
    sample_images = image_files[:5]

    # Visualize the images
    plt.figure(figsize=(15, 10))
    for i, image_name in enumerate(sample_images):
        # Read the image using Pillow
        image_path = os.path.join(dolphin_folder, image_name)
        image = Image.open(image_path)

        # Display the image
        plt.subplot(1, len(sample_images), i + 1)
        plt.imshow(image)
        plt.title(image_name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()