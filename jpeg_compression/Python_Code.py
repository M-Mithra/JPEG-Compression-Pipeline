# Import the required modules
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the quantization matrix
quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)
quantization_matrix_Y = quantization_matrix
quantization_matrix_CrCb = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float32,
)

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR using formula: PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)"""
    mse = np.mean((img1 - img2) ** 2)  # Calculate the Mean Squared Error (MSE) between the two images
    psnr = 20 * np.log10(255 / np.sqrt(mse))  # Compute PSNR using the maximum pixel intensity (255 for 8-bit images) and MSE
    return psnr  # Return the calculated PSNR value

def number_of_elements(blocks: list[np.ndarray]) -> int:
    """Calculates the total number of elements in the grayscale JPEG encoded array"""
    total_elements = 0  # Initialize the total element count to zero
    for block in blocks:  # Iterate through each block in the list
        # Trim the trailing zeros from the 1D array (helps reduce unnecessary zeros in compression)
        total_elements += np.trim_zeros(block, "b").size  # Add the size of the trimmed array to the total
    return total_elements  # Return the final count of total elements

def total_number_of_elements(blocks: list[np.ndarray]| tuple[list[np.ndarray],list[np.ndarray],list[np.ndarray],],color: bool,) -> int:
    """
    Calculates the total number of elements for both color and grayscale JPEG encoded arrays
    This is a utility function that will be used to calculate the compression ratio
    """
    total_elements = 0  # Initialize the total element count to zero
    if color:  # If the image is in color (has multiple channels)
        # Sum the number of elements across each color channel (e.g., R, G, B)
        total_elements = (number_of_elements(blocks[0]) + number_of_elements(blocks[1]) + number_of_elements(blocks[2]))
    else:  # If the image is grayscale (single channel)
        total_elements = number_of_elements(blocks)  # Count elements in the grayscale channel
    return total_elements  # Return the final count of total elements

def zigzag_scan(block: np.ndarray) -> np.ndarray:
    """
    Scans a block in zigzag order and returns a 1D array.
    Each block is assumed to be a square matrix.
    """
    a = 0  # Placeholder variable, likely unused (can be removed if not used elsewhere)
    block_size = block.shape[0]  # Determine the size of the square block (assumes block is square)
    # Generate a 1D array by scanning diagonals of the block in a zigzag pattern
    zigzag_arr = np.concatenate(
        [
            np.diagonal(block[::-1, :], i)[:: (2 * (i % 2) - 1)]  # Extract diagonals in zigzag order
            for i in range(1 - block_size, block_size)  # Iterate over possible diagonals in the block
        ]
    )
    return zigzag_arr  # Return the 1D array of elements in zigzag order

def zigzag_unscan(zigzag_arr: np.ndarray, block_size: int) -> np.ndarray:
    """Unscans a 1D array in zigzag order and returns a 2D array."""
    # Create an empty 2D array to store the unscanned values
    block = np.zeros((block_size, block_size), dtype=np.float32)  # Initialize a 2D array of the given block size
    x, y = 0, 0  # Initialize coordinates at the top-left corner
    for num in zigzag_arr:  # Iterate through each element in the 1D zigzag array
        block[x, y] = num  # Place the current element in the (x, y) position of the 2D array
        # Determine the direction to move based on the current position
        # If the sum of the coordinates is even, move in the "up-right" zigzag pattern
        if (x + y) % 2 == 0:
            # If at the last column, move down one row
            if y == block_size - 1:
                x += 1
            # If at the first row, move right one column
            elif x == 0:
                y += 1
            # Otherwise, move up one row and right one column
            else:
                x -= 1
                y += 1
        # If the sum of the coordinates is odd, move in the "down-left" zigzag pattern
        else:
            # If at the last row, move right one column
            if x == block_size - 1:
                y += 1
            # If at the first column, move down one row
            elif y == 0:
                x += 1
            # Otherwise, move down one row and left one column
            else:
                x += 1
                y -= 1

    return block  # Return the reconstructed 2D array

def grayscale_jpeg_encoder(img: np.ndarray, block_size: int, num_coefficients: int, quantization_matrix) -> list[np.ndarray]:
    # Pad the image to make its height and width divisible by the block size
    height, width = img.shape  # Get the height and width of the input image
    padded_height = height + (block_size - height % block_size) % block_size  # Calculate padded height
    padded_width = width + (block_size - width % block_size) % block_size  # Calculate padded width
    padded_img = np.zeros((padded_height, padded_width), dtype=np.uint8)  # Initialize padded image with zeros
    padded_img[:height, :width] = img  # Copy the original image into the padded image

    # Subtract 128 from the image to center pixel values around zero (JPEG color space adjustment)
    padded_img = padded_img.astype(np.float32) - 128  # Convert to float32 and subtract 128 for DC shift

    # Split the image into non-overlapping blocks of the given block size
    blocks = [
        padded_img[i : i + block_size, j : j + block_size]
        for i in range(0, padded_height, block_size)
        for j in range(0, padded_width, block_size)
    ]  # Generate blocks of the image for DCT processing

    # Apply the Discrete Cosine Transform (DCT) to each block to transform to frequency domain
    dct_blocks = [cv.dct(block) for block in blocks]  # DCT transformation for each block

    # Resize the quantization matrix to match the block size for consistency
    resized_quantization_matrix = cv.resize(
        quantization_matrix, (block_size, block_size), cv.INTER_CUBIC
    )  # Resize quantization matrix using cubic interpolation

    # Quantize each DCT coefficient by dividing each DCT value by the resized quantization matrix
    quantized_blocks = [
        np.round(block / resized_quantization_matrix).astype(np.int32)
        for block in dct_blocks
    ]  # Quantize each block by rounding the coefficients

    # Perform zigzag scanning on each quantized block to create a 1D array
    zigzag_scanned_blocks = [zigzag_scan(block) for block in quantized_blocks]  # Apply zigzag scan

    # Retain only the first `num_coefficients` coefficients in each zigzag-scanned block
    first_num_coefficients = [
        block[:num_coefficients] for block in zigzag_scanned_blocks
    ]  # Retain the specified number of coefficients

    return first_num_coefficients  # Return the list of quantized and zigzag-scanned coefficients

def grayscale_jpeg_decoder(blocks: list[np.ndarray], img: np.ndarray, block_size: int, quantization_matrix) -> np.ndarray:
    # Calculate the padded height and width of the image to handle non-multiple dimensions
    height, width = img.shape  # Get the height and width of the input image
    padded_height = height + (block_size - height % block_size) % block_size  # Calculate padded height
    padded_width = width + (block_size - width % block_size) % block_size  # Calculate padded width

    # Resize the quantization matrix to match the block size for consistency
    resized_quantization_matrix = cv.resize(
        quantization_matrix, (block_size, block_size), cv.INTER_CUBIC
    )  # Resize the quantization matrix using cubic interpolation to match block size

    # Unscan the zigzag-scanned blocks to recover the quantized DCT blocks
    zigzag_unscanned_blocks = [zigzag_unscan(block, block_size) for block in blocks]  # Reverse zigzag scanning

    # Dequantize the blocks by multiplying each by the resized quantization matrix
    dequantized_blocks = [
        block * resized_quantization_matrix for block in zigzag_unscanned_blocks
    ]  # Dequantize by element-wise multiplication with the quantization matrix

    # Apply the Inverse Discrete Cosine Transform (IDCT) to each dequantized block to recover spatial domain data
    idct_blocks = [cv.idct(block) for block in dequantized_blocks]  # Perform IDCT on each block

    # Reconstruct the compressed image from the IDCT blocks by placing them back in the original positions
    compressed_img = np.zeros((padded_height, padded_width), dtype=np.float32)  # Initialize the padded image array
    block_index = 0  # Index to track the current block
    for i in range(0, padded_height, block_size):  # Loop through rows
        for j in range(0, padded_width, block_size):  # Loop through columns
            compressed_img[i : i + block_size, j : j + block_size] = idct_blocks[
                block_index
            ]  # Place the IDCT block in the corresponding position
            block_index += 1  # Increment the block index

    # Add 128 to shift pixel values back to the original range
    compressed_img += 128  # Add 128 to reverse the DC shift from encoding

    # Clip the values to ensure they stay within the valid pixel range (0-255)
    compressed_img = np.clip(compressed_img, 0, 255)  # Clip pixel values to the range [0, 255]

    # Crop the image back to its original size and convert to uint8 for image format
    return compressed_img[:height, :width].astype(np.uint8)  # Return the cropped and properly formatted image

def color_jpeg_encoder(img: np.ndarray, block_size: int, num_coefficients: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    # Convert the image to YCbCr color space, which separates luminance (Y) and chrominance (Cb, Cr)
    ycbcr_image = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)  # Convert BGR image to YCbCr color space
    
    # Split the YCbCr image into separate Y, Cb, and Cr channels
    y_channel, cb_channel, cr_channel = cv.split(ycbcr_image)  # Split the YCbCr image into Y, Cb, Cr
    
    return (
        grayscale_jpeg_encoder(y_channel, block_size, num_coefficients, quantization_matrix_Y),  # Encode Y channel
        grayscale_jpeg_encoder(cb_channel, block_size, num_coefficients, quantization_matrix_CrCb),  # Encode Cb channel
        grayscale_jpeg_encoder(cr_channel, block_size, num_coefficients, quantization_matrix_CrCb),  # Encode Cr channel
    )

def color_jpeg_decoder(blocks: tuple[list[np.ndarray],list[np.ndarray],list[np.ndarray],],img: np.ndarray,block_size: int,) -> np.ndarray:
    # Convert the image to YCbCr color space for decoding
    ycbcr_image = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)  # Convert the input BGR image to YCbCr color space
    
    # Split the YCbCr image into separate Y, Cb, and Cr channels
    y_channel, cb_channel, cr_channel = cv.split(ycbcr_image)  # Split into Y, Cb, and Cr channels
    y_channel = grayscale_jpeg_decoder(blocks[0], y_channel, block_size, quantization_matrix_Y)  # Decode Y channel
    cb_channel = grayscale_jpeg_decoder(blocks[1], cb_channel, block_size, quantization_matrix_CrCb)  # Decode Cb channel
    cr_channel = grayscale_jpeg_decoder(blocks[2], cr_channel, block_size, quantization_matrix_CrCb)  # Decode Cr channel
    
    # Merge the decoded Y, Cb, and Cr channels back into a single YCbCr image
    ycbcr_decoded = cv.merge((y_channel, cb_channel, cr_channel))  # Merge the decoded channels into YCbCr image
    
    # Convert the YCbCr image back to BGR color space for display
    bgr_decoded = cv.cvtColor(ycbcr_decoded, cv.COLOR_YCrCb2RGB)  # Convert YCbCr to BGR for standard display
    return bgr_decoded

def jpeg_encoder(img_path: str,block_size: int,num_coefficients: int,color: bool,) -> (list[np.ndarray]| tuple[list[np.ndarray],list[np.ndarray],list[np.ndarray],]):
    if color:  # Check if the image is color or grayscale
        # Load the color image from the specified path
        img = cv.imread(img_path, cv.IMREAD_COLOR)  # Read the image as a color (BGR) image
        
        # Apply color JPEG encoding by passing the image to the color JPEG encoder
        return color_jpeg_encoder(img, block_size, num_coefficients)  # Return the JPEG encoded color image
        
    else:
        # Load the grayscale image from the specified path
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Read the image as a grayscale image
        
        # Apply grayscale JPEG encoding by passing the image to the grayscale JPEG encoder
        return grayscale_jpeg_encoder(img, block_size, num_coefficients, quantization_matrix)  # Return the JPEG encoded grayscale image

def jpeg_decoder(blocks: list[np.ndarray]| tuple[list[np.ndarray],list[np.ndarray],list[np.ndarray],],img_path: str,block_size: int,color: bool,) -> np.ndarray:
    if color:  # Check if the image is color or grayscale
        # Load the color image from the specified path
        img = cv.imread(img_path, cv.IMREAD_COLOR)  # Read the image as a color (BGR) image
        # Apply color JPEG decoding using the provided blocks and return the decoded image
        return color_jpeg_decoder(blocks, img, block_size)  # Decode the color JPEG image
        
    else:
        # Load the grayscale image from the specified path
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Read the image as a grayscale image
        # Apply grayscale JPEG decoding using the provided blocks and return the decoded image
        return grayscale_jpeg_decoder(blocks, img, block_size, quantization_matrix)  # Decode the grayscale JPEG image

def analyze_image(img_path: str, block_size: int, num_coefficients: int, color: bool) -> tuple[np.ndarray,np.ndarray,float,float,list[np.ndarray]| tuple[list[np.ndarray],list[np.ndarray],list[np.ndarray],],bool,]:
    # Read the image based on the color flag
    img: np.ndarray = None  # Initialize the image variable
    if color:  # If the image is in color
        img = cv.imread(img_path, cv.IMREAD_COLOR)  # Read the color image
    else:  # If the image is grayscale
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Read the grayscale image

    # Encode the image using JPEG compression
    encoded_img = jpeg_encoder(img_path, block_size, num_coefficients, color)  # Perform JPEG encoding

    # Decode the image using JPEG compression
    compressed_img = jpeg_decoder(encoded_img, img_path, block_size, color)  # Perform JPEG decoding

    # Calculate the Peak Signal-to-Noise Ratio (PSNR) between the original and compressed images
    psnr = cv.PSNR(img, compressed_img)  # PSNR is a metric to measure image quality loss

    # Calculate the compression ratio
    n2 = total_number_of_elements(encoded_img, color)  # Get the total number of elements in the encoded image
    if n2 == 0:  # If the total number of elements is 0, avoid division by zero
        compression_ratio = 0  # Set compression ratio to 0
    else:
        compression_ratio = img.size / total_number_of_elements(encoded_img, color)  # Compute compression ratio
    return (img, compressed_img, psnr, compression_ratio, encoded_img, color)

def save_compressed_image(compressed_img: np.ndarray,compression_ratio: float,) -> None:
    # Generate the filename for the compressed image, using the compression ratio
    compressed_img_filename = f"compressed_{compression_ratio:.2f}.jpeg"

    # Display the compressed image using matplotlib
    plt.imshow(compressed_img, cmap="gray")  # Assuming grayscale image, use appropriate colormap if necessary
    plt.title(f"Compressed Image - Compression Ratio = {compression_ratio:.2f}")  # Set the title with compression ratio
    plt.axis("off")  # Turn off axis labels to focus on the image itself
    plt.savefig(compressed_img_filename)  # Save the image as a .jpeg file with the computed filename
    plt.show()  # Show the image in the plot

def plot_images(img: np.ndarray,compressed_img: np.ndarray,psnr: float,compression_ratio: float,encoded_img: list[np.ndarray]| tuple[list[np.ndarray],list[np.ndarray],list[np.ndarray],],color: bool,) -> None:
    # Create a subplot with 1 row and 2 columns for displaying the images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Set the title of the whole figure to display the compression ratio
    fig.suptitle(
        "Compression Ratio = {:.2f}".format(compression_ratio)
    )
    print(f"compression : {compression_ratio}")  # Print the compression ratio for debugging
    
    # Generate a filename for saving the compressed image based on the compression ratio
    compressed_img_filename = f"compressed_{compression_ratio:.2f}.jpeg"
    
    # Open a text file to write the encoded image data
    with open("encoded_image.txt", "w") as f:
        if color:
            # If the image is in color, convert from BGR to RGB before displaying
            axs[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # Display the original image
            axs[1].imshow(cv.cvtColor(compressed_img, cv.COLOR_BGR2RGB))  # Display the compressed image
            
            # Write the encoded image data to the text file
            for row in zip(*encoded_img):
                for element in row:
                    f.write(str(element) + " ")
                f.write("\n")
        else:
            # If the image is grayscale, display with a gray colormap
            axs[0].imshow(img, cmap="gray")  # Display the original grayscale image
            axs[1].imshow(compressed_img, cmap="gray")  # Display the compressed grayscale image
            
            # Write the encoded image data to the text file
            for row in encoded_img:
                for element in row:
                    f.write(str(element) + " ")
                f.write("\n")
    # Set titles for each subplot (original and compressed images)
    axs[0].set_title("Original Image")
    axs[1].set_title("Compressed Image")

    # Display the images and the compression results
    plt.show()

def plot_graph(img_dir_path: str,color: bool,):
    psnr_list = []  # List to store the average PSNR values for different coefficients
    compression_ratio_list = []  # List to store the average compression ratios for different coefficients
    
    # Loop through different numbers of coefficients to analyze their effect on compression
    for num_coefficients in [1, 3, 6, 10, 15, 28]:
        psnr_values = []  # Temporary list to store PSNR values for each image
        compression_ratio_values = []  # Temporary list to store compression ratios for each image
        
        # Loop through all images in the specified directory
        for img_file in os.listdir(img_dir_path):
            img_path = os.path.join(img_dir_path, img_file)  # Get full path of the image file
            
            # Analyze the image by performing JPEG compression and retrieving relevant data
            _, _, psnr, compression_ratio, _, _ = analyze_image(
                img_path, 8, num_coefficients, color
            )
            
            # Append the PSNR and compression ratio for this image to their respective lists
            psnr_values.append(psnr)
            compression_ratio_values.append(compression_ratio)
        
        # Compute the average PSNR and compression ratio across all images for the current coefficient count
        psnr_list.append(np.mean(psnr_values))
        compression_ratio_list.append(np.mean(compression_ratio_values))

    # Plot the average PSNR vs Compression Ratio on a scatter plot
    plt.plot(compression_ratio_list, psnr_list, "o")
    plt.xlabel("Compression Ratio")
    plt.ylabel("PSNR")
    plt.title("PSNR vs Compression Ratio")
    plt.show()

if __name__ == "__main__":  # Check if this script is being run directly (not imported as a module)
    # Prompt user if they want to analyze a single image
    if input("Analyze a single image (y/n): ") == "y":  
        # If user chooses to analyze a single image
        img_path = input("Enter the path to the image: ")  # Ask for the image path
        block_size = int(input("Enter the block size (even): "))  # Ask for the block size (should be even)
        num_coefficients = int(input("Enter the number of coefficients passed: "))  # Ask for number of coefficients
        color = input("Is the image color (y/n): ") == "y"  # Ask if the image is color (convert to boolean)
        
        # Analyze the image and plot the original and compressed images
        plot_images(*analyze_image(img_path, block_size, num_coefficients, color))

    # Prompt user if they want to analyze all images in a folder
    elif input("Analyzes all images in a folder (y/n): ") == "y":  
        # If user chooses to analyze all images in a folder
        img_dir_path = input("Enter the path to the images folder: ")  # Ask for the path to the folder containing images
        color = input("Are the images color (y/n): ") == "y"  # Ask if the images are color (convert to boolean)
        
        # Generate and display the PSNR vs Compression Ratio graph for all images in the folder
        plot_graph(img_dir_path, color)  # Call plot_graph to analyze all images in the folder