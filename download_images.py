import os
import requests

# Base URL of the images
base_url = "https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/world_500m/"

# List of image names to download
image_names = [
    "world.200401.3x21600x21600.A1.jpg",
    "world.200401.3x21600x21600.A2.jpg",
    "world.200401.3x21600x21600.B1.jpg",
    "world.200401.3x21600x21600.B2.jpg",
    "world.200401.3x21600x21600.C1.jpg",
    "world.200401.3x21600x21600.C2.jpg",
    "world.200401.3x21600x21600.D1.jpg",
    "world.200401.3x21600x21600.D2.jpg"
]

# Directory to save downloaded images
download_dir = r"C:\Users\chris\OneDrive\Work\2024\CSC4002W Project\Data\86400x43200 JPG"

# Function to download an image
def download_image(image_name):
    image_url = base_url + image_name
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        file_path = os.path.join(download_dir, image_name)
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Downloaded {image_name}")
    else:
        print(f"Failed to download {image_name}")

# Download all images
for image_name in image_names:
    download_image(image_name)
