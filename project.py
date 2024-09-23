# Question 1
import cv2
from PIL import Image
import numpy as np

image = Image.open('Images/lena.tif')
width, height = image.size

# Calculate new dimensions (1/2 of the original in each dimension)
down_width = width // 2
down_height = height // 2

downscaled_image = image.resize((down_width, down_height), Image.LANCZOS)
downscaled_image_np = np.array(downscaled_image)

# Create an empty array for the rescaled image (original dimensions)
rescaled_image_np = np.zeros((height, width, 3), dtype=downscaled_image_np.dtype)

# Nearest-neighbor interpolation
for i in range(height):
    for j in range(width):
        # Find the nearest pixel in the downscaled image
        nearest_i = int(i / 2)
        nearest_j = int(j / 2)

        # Assign the nearest pixel value to the rescaled image
        rescaled_image_np[i, j] = downscaled_image_np[nearest_i, nearest_j]

n_neighbor_scratch_image = Image.fromarray(rescaled_image_np)
n_neighbor_scratch_image.save('Results/Question 1/lena_nearest_scratch.tif')

# Convert the downscaled image to OpenCV format
downscaled_image_cv = cv2.cvtColor(np.array(downscaled_image), cv2.COLOR_RGB2BGR)

# Use OpenCV's resize function with INTER_NEAREST for nearest neighbor interpolation
upscaled_image_cv = cv2.resize(downscaled_image_cv, (width, height), interpolation=cv2.INTER_NEAREST)

n_neighbor_cv_image = Image.fromarray(cv2.cvtColor(upscaled_image_cv, cv2.COLOR_BGR2RGB))
n_neighbor_cv_image.save('Results/Question 1/lena_nearest_cv.tif')

downscaled_array = np.array(downscaled_image)

# Initialize an empty array for the upscaled image
upscaled_array = np.zeros((height, width), dtype=downscaled_array.dtype)

# Calculate scaling factors
scale_x = down_width / width
scale_y = down_height / height

# Perform bilinear interpolation
for i in range(height):
    for j in range(width):
        # Corresponding position in the downscaled image
        x = j * scale_x
        y = i * scale_y

        # Coordinates of the top-left pixel
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))

        # Coordinates of the bottom-right pixel
        x1 = min(x0 + 1, down_width - 1)
        y1 = min(y0 + 1, down_height - 1)

        # Differences
        dx = x - x0
        dy = y - y0

        # Retrieve pixel values
        Ia = downscaled_array[y0, x0]
        Ib = downscaled_array[y1, x0]
        Ic = downscaled_array[y0, x1]
        Id = downscaled_array[y1, x1]

        # Compute interpolated value
        value = (1 - dx) * (1 - dy) * Ia + dx * (1 - dy) * Ic + (1 - dx) * dy * Ib + dx * dy * Id
        upscaled_array[i, j] = np.clip(value, 0, 255)

upscaled_image = Image.fromarray(upscaled_array.astype(np.uint8), mode='L')
upscaled_image.save('Results/Question 1/lena_bilinear_scratch.tif')

# Use OpenCV's resize function with INTER_LINEAR for bilinear interpolation
upscaled_image_cv = cv2.resize(downscaled_image_cv, (width, height), interpolation=cv2.INTER_LINEAR)

bilinear_cv_image = Image.fromarray(cv2.cvtColor(upscaled_image_cv, cv2.COLOR_BGR2RGB))
bilinear_cv_image.save('Results/Question 1/lena_bilinear_cv.tif')

# Use OpenCV's resize function with INTER_CUBIC for bicubic interpolation
upscaled_image_cv = cv2.resize(downscaled_image_cv, (width, height), interpolation=cv2.INTER_CUBIC)

bicubic_cv_image = Image.fromarray(cv2.cvtColor(upscaled_image_cv, cv2.COLOR_BGR2RGB))
bicubic_cv_image.save('Results/Question 1/lena_bicubic_cv.tif')

def compute_mse(image1_path, image2_path):
    """
    Computes the Mean Squared Error between two images.

    Parameters:
        image1_path (str): Path to the first image (original image).
        image2_path (str): Path to the second image (interpolated image).

    Returns:
        float: The MSE between the two images.
    """
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')

    img1_np = np.array(img1, dtype=np.float64)
    img2_np = np.array(img2, dtype=np.float64)

    # Ensure the images have the same dimensions
    if img1_np.shape != img2_np.shape:
        raise ValueError(f"Image dimensions do not match: {img1_np.shape} vs {img2_np.shape}")

    # Compute the Mean Squared Error
    mse = np.mean((img1_np - img2_np) ** 2)
    return mse

resultsPath = 'Results/Question 1/'
original_image_path = 'Images/lena.tif'

interpolated_images = {
    'Nearest Neighbor (Scratch)': f'{resultsPath}lena_nearest_scratch.tif',
    'Nearest Neighbor (OpenCV)': f'{resultsPath}lena_nearest_cv.tif',
    'Bilinear (Scratch)': f'{resultsPath}lena_bilinear_scratch.tif',
    'Bilinear (OpenCV)': f'{resultsPath}lena_bilinear_cv.tif',
    'Bicubic (Scratch)': f'{resultsPath}lena_bicubic_scratch.tif',
    'Bicubic (OpenCV)': f'{resultsPath}lena_bicubic_cv.tif'
}

# Compute and print the MSE for each interpolated image
print("Mean Squared Error between Original and Interpolated Images:")
for method, image_path in interpolated_images.items():
    try:
        mse = compute_mse(original_image_path, image_path)
        print(f"{method}: MSE = {mse:.2f}")
    except Exception as e:
        print(f"{method}: Error - {e}")

# Question 2
image_path = 'Images/cameraman.tif'
img = Image.open(image_path).convert('L')
img_np = np.array(img)

# Negative of the Image
def negative_image(image_np):
    return 255 - image_np

# Save the negative image
resultsPath = 'Results/Question 2/'
cameraman_negative = negative_image(img_np)
Image.fromarray(cameraman_negative).save(f'{resultsPath}cameraman_negative.tif')

# Power-Law Transformation (Gamma Correction)
def power_law_transformation(image_np, gamma, c=1):
    # Normalize the image to the range [0, 1]
    normalized_img = image_np / 255.0

    # Apply the power-law transformation
    transformed_img = c * np.power(normalized_img, gamma)

    # Scale back to [0, 255] and cast to uint8
    return np.uint8(transformed_img * 255)

gamma_value = 1.5
cameraman_power = power_law_transformation(img_np, gamma_value)
Image.fromarray(cameraman_power).save(f'{resultsPath}cameraman_power.tif')

# Bit-Plane Slicing
def bit_plane_slicing(image_np, bit):
    # Shift the bits and apply bitwise AND with 1 to extract the bit-plane
    return (image_np >> bit) & 1

# Save bit-plane images
for i in range(8):
    bit_plane_image = bit_plane_slicing(img_np, i) * 255  # Scale to [0, 255]
    Image.fromarray(np.uint8(bit_plane_image)).save(f'{resultsPath}cameraman_b{i+1}.tif')

# Question 3
image_path = 'Images/einstein.tif'
img = Image.open(image_path).convert('L')
img_np = np.array(img)

# Compute the histogram
hist, bins = np.histogram(img_np.flatten(), bins=256, range=[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * 255 / cdf[-1]  # Scale to [0,255]

# Use linear interpolation of the CDF to find new pixel values
img_equalized_np = np.interp(img_np.flatten(), bins[:-1], cdf_normalized)

# Reshape the flattened array back to the original image shape
img_equalized_np = img_equalized_np.reshape(img_np.shape).astype('uint8')

img_equalized = Image.fromarray(img_equalized_np)
img_equalized.save('Results/Question 3/einstein_equalized.tif')

def histogram_matching(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image.

    Parameters:
    - source: NumPy array of the source image
    - template: NumPy array of the template image (reference)

    Returns:
    - matched: NumPy array of the transformed output image
    """
    # Flatten the images
    source_flat = source.ravel()
    template_flat = template.ravel()

    # Get the set of unique pixel values and their corresponding indices and counts
    _, bin_idx, s_counts = np.unique(source_flat, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template_flat, return_counts=True)

    # Compute the normalized CDFs
    s_quantiles = np.cumsum(s_counts).astype(np.float64) / source_flat.size
    t_quantiles = np.cumsum(t_counts).astype(np.float64) / template_flat.size

    # Create interpolation function (inverse CDF of the template)
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    # Map the pixel values of the source image to the template
    matched = interp_t_values[bin_idx].reshape(source.shape).astype('uint8')

    return matched

# Load the source images
source_image_path = 'Images/chest_x-ray1.jpeg'
template_image_path = 'Images/chest_x-ray2.jpeg'

source_img = Image.open(source_image_path).convert('L')
template_img = Image.open(template_image_path).convert('L')

source_np = np.array(source_img)
template_np = np.array(template_img)

# Apply histogram matching
matched_np = histogram_matching(source_np, template_np)

# Save the output image
matched_img = Image.fromarray(matched_np)
matched_img.save('Results/Question 3/chest_x-ray3.jpeg')