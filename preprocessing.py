import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def dullrazor(image_path, output_path=None):
    """
    Robust DullRazor using directional morphological line filters.
    Highly effective for both fine and thick hairs in dermoscopy.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Blackhat filtering in multiple directions (0, 45, 90, 135 degrees)
    kernel_length = 17  
    
    # Create empty image to store the max hair response
    max_blackhat = np.zeros_like(gray)
    
    # Loop through different angles to catch hairs going in any direction
    for angle in [0, 45, 90, 135]:
        kernel = np.zeros((kernel_length, kernel_length), dtype=np.uint8)
        center = kernel_length // 2
        
        if angle == 0:
            kernel[center, :] = 1
        elif angle == 90:
            kernel[:, center] = 1
        elif angle == 45:
            np.fill_diagonal(kernel, 1)
        elif angle == 135:
            np.fill_diagonal(np.fliplr(kernel), 1)
            
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        max_blackhat = cv2.max(max_blackhat, blackhat)
    
    # 3. Thresholding to create mask (Pixels brighter than 10 are hair)
    _, hair_mask = cv2.threshold(max_blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 4. Dilate the mask slightly to cover the soft edges of the hairs
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_mask = cv2.dilate(hair_mask, kernel_dilate, iterations=1)
    
    # 5. Inpaint the hairs
    cleaned = cv2.inpaint(img, hair_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    if output_path:
        cv2.imwrite(output_path, cleaned)
    
    return cleaned, hair_mask


def apply_clahe(img):
    """
    Applies CLAHE safely to a dermoscopy image.
    - Works in LAB color space to avoid color distortion
    - Uses clipLimit=2.0 to avoid amplifying DullRazor artifacts
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    return final_image


def visualize_pipeline(image_path):
    """Shows Original → After DullRazor → After DullRazor + CLAHE"""
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Process image
    after_dullrazor, _ = dullrazor(image_path)
    after_clahe = apply_clahe(after_dullrazor)

    # Convert to RGB for matplotlib
    imgs = [original, after_dullrazor, after_clahe]
    titles = ["Original", "After DullRazor", "After DullRazor + CLAHE"]
    imgs_rgb = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, title in zip(axes, imgs_rgb, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=13)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("clahe_comparison.png", dpi=150)
    plt.show()
    print("Saved comparison to clahe_comparison.png")


def process_entire_dataset(input_folder, output_folder):
    """
    Batch processing function for the desktop (GPU/CPU) runs.
    Applies DullRazor + CLAHE to all images in a folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    print(f"Found {len(image_files)} images to process.")
    failed_images = []

    for filename in tqdm(image_files, desc="Preprocessing"):
        input_path  = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.exists(output_path):
            continue

        try:
            cleaned, _ = dullrazor(input_path)
            final = apply_clahe(cleaned)
            cv2.imwrite(output_path, final)
        except Exception as e:
            failed_images.append((filename, str(e)))

    print("\n--- Processing Complete ---")
    if failed_images:
        print(f"Failed to process {len(failed_images)} images.")
    else:
        print("All images processed successfully!")


if __name__ == "__main__":
    # ── TEST ON YOUR LAPTOP FIRST ────────────────────────
    test_image = r"data\ISIC_0000095.jpg"
    print("Running CLAHE pipeline test on single image...")
    visualize_pipeline(test_image)

    # ── DESKTOP BATCH RUN (Uncomment when ready) ──────────
    # INPUT_DIR  = r"C:\Users\astro\Desktop\lesioniq\preprocesing\ISIC_2019_Training_Input\ISIC_2019_Training_Input"
    # OUTPUT_DIR = r"C:\Users\astro\Desktop\lesioniq\preprocesing\processed"
    # print("-" * 50)
    # process_entire_dataset(INPUT_DIR, OUTPUT_DIR)