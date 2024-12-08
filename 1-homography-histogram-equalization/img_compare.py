import os
from PIL import Image
import numpy as np

def compare_images(png_path1, png_path2, output_mask_path, output_subtracted_path):
    # Open the PNG files
    try:
        image1 = Image.open(png_path1)
        image2 = Image.open(png_path2)
    except Exception as e:
        print(f"Error opening images: {e}")
        return

    # Ensure both images are of the same size
    if image1.size != image2.size:
        print(f"Image sizes differ: Image 1 size {image1.size}, Image 2 size {image2.size}")
        return
    
    if image2.mode == 'P':  # Check if the image is palettized
        image2 = image2.convert('RGB')

    # Convert images to NumPy arrays
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # Handle grayscale images
    if len(image1_array.shape) == 2:  # Grayscale image 1
        image1_array = np.stack([image1_array] * 3, axis=-1)  # Convert to RGB by stacking channels
    if len(image2_array.shape) == 2:  # Grayscale image 2
        image2_array = np.stack([image2_array] * 3, axis=-1)  # Convert to RGB by stacking channels

    # Ensure the shapes match after conversion
    if image1_array.shape != image2_array.shape:
        print(f"Image shapes differ after conversion: Image 1 shape {image1_array.shape}, Image 2 shape {image2_array.shape}")
        return

    # Calculate the difference between the two images
    difference = np.abs(image1_array.astype(int) - image2_array.astype(int))

    # Create a binary mask of the differences (0 where equal, non-zero where different)
    mask = np.sum(difference, axis=-1) > 0

    if np.sum(mask) == 0:
        print("The images are identical.")
        return
    else:
        print(f"{np.sum(mask)} pixels differ between the images.")

    # 1. Generate the red mask highlighting discrepancies
    diff_image = image2_array.copy()
    diff_image[mask] = [255, 0, 0]  # Red color for discrepancies
    diff_output_image = Image.fromarray(diff_image.astype('uint8'))
    diff_output_image.save(output_mask_path)
    print(f"Difference image saved at {output_mask_path}")

    # 2. Generate the subtracted image for detailed analysis
    subtracted_image = np.clip((image1_array.astype(int) - image2_array.astype(int)) * 1, 0, 255)  # Clip values between 0 and 255
    subtracted_output_image = Image.fromarray(subtracted_image.astype('uint8'))
    subtracted_output_image.save(output_subtracted_path)
    print(f"Subtracted image saved at {output_subtracted_path}")

def main():
    # File paths for Eric and Miche images
    png_file1 = 'images/eric.png'
    png_file2 = 'images/miche.png'
    output_mask_file = 'validate/eric_vs_miche_diff.png'  # Red mask highlighting differences
    output_subtracted_file = 'validate/eric_vs_miche_subtracted.png'  # Raw subtracted image

    # Ensure output directory exists
    os.makedirs('validate', exist_ok=True)

    # Compare the images and generate outputs
    compare_images(png_file1, png_file2, output_mask_file, output_subtracted_file)

if __name__ == '__main__':
    main()
