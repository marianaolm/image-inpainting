import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage
import skimage.morphology as morpho

### Functions ###

def select_points(event, x, y, flags, param):
    """
    Callback function to select points on the image.

    Parameters:
    - event: The event type (mouse action).
    - x: The x-coordinate of the mouse event.
    - y: The y-coordinate of the mouse event.
    - flags: Any relevant flags (not used).
    - param: Any additional parameters (not used).

    This function allows the user to click on the image to select points that 
    form a contour. The selected points are stored in the global 'points' list,
    and the image is updated to show the selected points and lines connecting them.
    When the user clicks on the image, a red circle is drawn at the clicked 
    position and green lines connect the selected points.
    """
    global points, img_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        img_copy = image.copy()

        for i, point in enumerate(points):
            cv2.circle(img_copy, point, 1, (0, 0, 255), -1)

            if i > 0:
                cv2.line(img_copy, points[i - 1], points[i], (0, 255, 0), 2)

        cv2.imshow('Image', img_copy)


def get_matrix_mask(image, points):
    # Create the contour array from the provided points
    contour = np.array(points, dtype=np.int32)  # Remove the extra array wrapping

    # Create a mask with the same size as the input image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Fill the polygonal area in the mask with the value 1
    cv2.fillPoly(mask, [contour], 1)  # Pass a list of contours

    # Create a binary mask where the region inside the polygon is 1, and outside is 0
    binary_mask = (mask > 0).astype(np.uint8)

    # Find the contours of the filled polygonal region
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Initialize contour_sets
    contour_sets = []

    if contours:
        first_contour = contours[0]
        
        # Convert contour points to a list of tuples (x, y)
        contour_sets = [(point[0][0], point[0][1]) for point in first_contour]

    return binary_mask, contour_sets

def get_fill_front(mask): 
    """
    Parameters:
    - binary_mask (numpy.ndarray): The binary mask with 1's for the target region and 0's for the rest.
    
    Returns the matrix with 1's in the position of the pixels in the fill front, and its coordenates.
    """
    erosion = morpho.binary_erosion(mask, morpho.square(3)) 
    border = mask - erosion
    border_indices = np.where(border == 1)

    return border, list(zip(border_indices[0], border_indices[1]))


def get_matrix_image(image, binary_mask):
    """
    Modifies the image based on a provided binary mask, highlighting the target region.

    This function takes an input image and a binary mask, and returns an image where 
    the target region (defined by 1's in the mask) is set to white (255), while the 
    rest of the image retains its original values.

    Parameters:
    - image (numpy.ndarray): The input image.
    - binary_mask (numpy.ndarray): The binary mask with 1's for the target region and 0's for the rest.

    Returns:
    - image_with_target_region (numpy.ndarray): Image where the target region is highlighted.
    """
    
    image_with_target_region = np.ones_like(image) * 255 
    image_with_target_region[binary_mask == 0] = image[binary_mask == 0]

    return image_with_target_region


def extract_window(pixel, window_size, matrix, binary_mask):
    """
    Extracts the coordinates of the pixels in a window around a given pixel
    and retrieves the corresponding values from the given matrix, considering a binary mask.
    
    Parameters:
    pixel (tuple): A tuple (x, y) representing the coordinates of the central pixel.
    window_size (int): The size of the window (must be an odd number for a symmetric patch).
    matrix (np.ndarray): A 2D array representing the matrix from which to extract values.
    binary_mask (np.ndarray): A binary mask of the same shape as `matrix`, where 1 indicates
                              positions to be treated as NaN in `matrix`.
    
    Returns:
    tuple: A tuple containing two elements:
        - list: A list of tuples representing the coordinates of the pixels in the window around the given pixel.
        - list: A list of values from the matrix corresponding to the extracted window coordinates,
                with NaN where the binary_mask has 1s.
    """
    x, y = pixel
    half_window = window_size // 2

    # Define the effective matrix by setting NaN where binary_mask is 1
    effective_matrix = np.where(binary_mask == 1, np.nan, matrix)
    
    # Create ranges for rows and columns within the window
    row_range = np.arange(max(0, x - half_window), min(matrix.shape[0], x + half_window + 1))
    col_range = np.arange(max(0, y - half_window), min(matrix.shape[1], y + half_window + 1))
    
    # Create a grid of coordinates using NumPy's meshgrid
    row_coords, col_coords = np.meshgrid(row_range, col_range, indexing='ij')
    
    # Flatten the coordinates to match the expected output format
    window_coordinates = list(zip(row_coords.ravel(), col_coords.ravel()))
    window_values = effective_matrix[row_coords, col_coords].ravel().tolist()
    
    return window_coordinates, window_values


def calculate_confidence(confidence_matrix, window_coordinates):
    """
    Calculate the confidence value based on the surrounding patch defined by window coordinates.
    
    Parameters:
    confidence_matrix (np.ndarray): A matrix representing the confidence values of each pixel.
    window_coordinates (list): A list of tuples representing the coordinates of the pixels in the window.
    
    Returns:
    float: The calculated confidence value as the average of the values in the window.
    """

    sum_confidence = 0.0
    
    for (x, y) in window_coordinates:
        sum_confidence += confidence_matrix[x, y]

    confidence_value = sum_confidence / len(window_coordinates)
    
    return confidence_value

def calculate_data(image, mask, p, d, sigma): 
    """
    Calculates the data term associated with a pixel.
    D = norm of the scalar produt between the orthogonal of the gradient of the image in the point p 
    and the normal of the target region border in the point p, all divided by alpha

    Parameters:
    - image (numpy.ndarray): The unfilled image. Has to be the image we are filling which is different from the original
    - mask (numpy.ndarray): The binary mask with 1's for the target region and 0's for the rest. 
    - d: dimension of the region to consider for the aproximation calculation. 
    - sigma: standard deviation of the gaussian kernell

    Returns:
    int: data term
    """
    gradient = calculate_gradient(image, mask, p, d, sigma)
    orthogonal = (-gradient[1], gradient[0])
    normal = calculate_normal(mask, p, d)
    alpha = 200  

    data = abs(orthogonal @ normal) / alpha

    return data
    
# def calculate_gradient(image_unfilled, mask, p, d, sigma): # without worring about begin clode to the border of the image
#     image = cv2.medianBlur(image_unfilled, ksize=3)

#     mask = 255 - mask
#     region = image[max(p[0]-d, 0) : min(p[0]+d, mask.shape[0]-1)+1, max(p[1]-d, 0) : min(p[1]+d, mask.shape[1]-1)+1]
#     region_mask = mask[max(p[0]-d, 0) : min(p[0]+d, mask.shape[0]-1)+1, max(p[1]-d, 0) : min(p[1]+d, mask.shape[1]-1)+1]

#     grad_x_region = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
#     grad_y_region = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)

#     kernel_1d = cv2.getGaussianKernel(2*d+1, sigma) 
#     kernel = np.outer(kernel_1d, kernel_1d)

#     kernel = kernel * region_mask
#     kernel = kernel / np.sum(kernel)

#     grad_x = np.sum(kernel * abs(grad_x_region))
#     grad_y = np.sum(kernel * abs(grad_y_region))

#     return (grad_x, grad_y)


def calculate_gradient(image_unfilled, mask, p, d, sigma, filter = False):
    """
    Calculates the gradient of the image at a point p of the target region border. The estimation is done over a region around p.
    Using a gaussian kernell to weight their contribution.

    Parameters:
    - image (numpy.ndarray): The input image.
    - mask (numpy.ndarray): The binary mask with 1's for the target region and 0's for the rest. 
    - p (tuple): Tuple with the coordenates of the point in which we calculate the data term.
    - d: dimension of the region to consider for the aproximation calculation. Has to be higher than 1, so there is no division by 0.
    - sigma: standard deviation of the gaussian kernell

    Returns:
    tuple: Vector (x, y) representing the gradient
    """
    image = image_unfilled
    if filter:
        image = cv2.medianBlur(image_unfilled, ksize=3)

    mask_dilated = morpho.binary_dilation(mask, morpho.square(3))
    mask_dilated = 1 - mask_dilated

    region = image[max(p[0]-d, 0) : min(p[0]+d, mask.shape[0]-1)+1, max(p[1]-d, 0) : min(p[1]+d, mask.shape[1]-1)+1]
    region_mask = mask_dilated[max(p[0]-d, 0) : min(p[0]+d, mask.shape[0]-1)+1, max(p[1]-d, 0) : min(p[1]+d, mask.shape[1]-1)+1]

    grad_x_region = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_region = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)

    x_i = max(p[0]-d, 0)
    x_f = min(p[0]+d, mask.shape[0]-1)
    y_i = max(p[1]-d, 0)
    y_f = min(p[1]+d, mask.shape[1]-1)

    x_left = p[0] - x_i
    x_right = x_f - p[0]
    y_up = p[1] - y_i
    y_down = y_f - p[1]

    # Gaussian kernel
    kernel_1d = cv2.getGaussianKernel(2*d+1, sigma) 
    kernel = np.outer(kernel_1d, kernel_1d)
    
    p_k = (d, d)

    kernel = kernel[p_k[0] - x_left : p_k[0] + x_right + 1, p_k[1] - y_up : p_k[1] + y_down + 1]
    kernel = kernel * region_mask
    kernel = kernel / np.sum(kernel)

    grad_x = np.sum(kernel * grad_x_region)
    grad_y = np.sum(kernel * grad_y_region)

    return (grad_x, grad_y)

def calculate_normal(mask, p, d):
    """
    Returns the vector normal to the border of the target region in the point p. Norm equals 1. 
    The estimation is done calculating over a square around p, summing the gradient of all pixels in that area.

    Parameters:
    - binary_mask (numpy.ndarray): The binary mask with 1's for the target region and 0's for the rest.
    - p (tuple): Tuple with the coordenates of the point in which we calculate the normal
    - d (int): "Window size" around p for which we consider to estimate the normal 

    Returns:
    tuple: Vector (x, y) of unitary norm 

    """
    mask = 1 - mask

    right_col = mask[max(p[0]-d, 0) : min(p[0]+d+1, mask.shape[0]), min(p[1]+d, mask.shape[1]-1)]
    left_col = mask[max(p[0]-d, 0) : min(p[0]+d+1, mask.shape[0]), max(p[1]-d, 0)]

    gradx = sum(right_col - left_col)

    bottom_row = mask[min(p[0]+d, mask.shape[0]-1), max(p[1]-d, 0) : min(p[1]+d, mask.shape[1]-1)+1]
    top_row = mask[max(p[0]-d, 0), max(p[1]-d, 0) : min(p[1]+d, mask.shape[1]-1)+1]

    grady = sum(bottom_row - top_row)

    if gradx == 0 and grady == 0:
        normal = np.array([0, 0])
    else:
        normal = np.array([gradx, grady]) / np.sqrt(gradx**2 + grady**2)

    return normal


def calculate_priorities(confidence: np.ndarray, data: np.ndarray):
    """
    Calculates the priority values for each pixel in the fill front.

    The function takes two vectors as input: 'confidence' and 'data', which correspond to the confidence values
    and data terms for the pixels in the fill front, respectively. It computes the priority for each pixel using
    the formula P = C * D, where C is the confidence value and D is the data term.

    Parameters:
    confidence (np.ndarray): A vector of confidence values for the pixels in the fill front.
    data (np.ndarray): A vector of data terms for the pixels in the fill front.

    Returns:
    int: The index of the pixel with the highest priority.
    """

    # Compute priority values
    priority_values = confidence * data

    # Find the index of the maximum priority value
    index_max_priority = np.argmax(priority_values)

    return index_max_priority


def extract_patches(image, window_size):
    N, S = image.shape
    m = window_size
    
    num_patches = (N - m + 1) * (S - m + 1)
    patches = np.zeros((num_patches, m * m))
    
    patch_views = np.lib.stride_tricks.sliding_window_view(image, (m, m))
    
    patches = patch_views.reshape((num_patches, m * m))

    coords = [(i, j) for i in range(N - m + 1) for j in range(S - m + 1)]

    return patches, coords


def patch_matching(image_unfilled, loaded_mask, patch_values, window_size):
    """
    This function identifies the optimal fill values for unfilled areas of an image 
    using the sum of squared differences (SSD) between patches from the unfilled image 
    and given patch values, ignoring areas marked as NaN in patch_values and avoiding 
    patches that contain unfilled regions according to loaded_mask.

    Returns:
    - fill_values (numpy.ndarray): The optimal fill values identified from the patches.
    - chosen_coords (tuple): The coordinates (row, col) of the top-left corner of the selected patch.
    """
    # Extract patches and their coordinates
    image_patches, image_coords = extract_patches(image_unfilled, window_size)
    loaded_mask_patches, mask_coords = extract_patches(loaded_mask, window_size)

    image_patches_array = np.array(image_patches)
    patch_values_array = np.array(patch_values)

    # Create a mask for NaN values in patch_values
    valid_mask = ~np.isnan(patch_values_array)  # True where values in patch_values are not NaN

    # Filter out any patches that correspond to unfilled areas in the loaded_mask
    loaded_mask_patches_array = np.array(loaded_mask_patches)
    valid_patches_mask = np.all(loaded_mask_patches_array == 0, axis=1)  # Only fully filled patches
    valid_image_patches = image_patches_array[valid_patches_mask]
    valid_coords = [image_coords[i] for i, valid in enumerate(valid_patches_mask) if valid]

    # Compute SSD only over valid (non-NaN) values
    diff = (valid_image_patches - patch_values_array) * valid_mask  # Ignores NaNs
    diff_squared = np.nan_to_num(diff, nan=0) ** 2
    ssd_vector = np.sum(diff_squared, axis=1)  # Sum over valid positions only

    # Identify the index of the patch with the minimum SSD
    min_index = np.argmin(ssd_vector)

    # Return the optimal fill values and the corresponding coordinates
    fill_values = valid_image_patches[min_index]
    chosen_coords = valid_coords[min_index]

    return fill_values, chosen_coords


def create_test_csv(image, csv_name):
    points = []
    base_dir = os.path.dirname(os.path.abspath(__file__))

    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(image, points[-2], points[-1], (255, 0, 0), 1)
            cv2.imshow('Image', image)

    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', select_points)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if len(points) >= 3:
        binary_mask, _ = get_matrix_mask(image.copy(), points)

        output_dir = os.path.join(base_dir, 'test_files')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{csv_name}.csv')

        np.savetxt(output_path, binary_mask, delimiter=",")

    cv2.destroyAllWindows()
 


### Main ###

if __name__ == "__main__":
    drawing = False
    points = []

    window_size = 11
    output_dir = "test/iteration_images"
    video_path = "test/video/filled_image_video_aerian11.mp4"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Load the image
    image = cv2.imread('images/aerien1.jpg', cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread('images/chile.png', cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread('images/shapes_image.png', cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread('images/bateau.jpg', cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread('images/duck_256.jpg', cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread('images/bergamo_big.JPG', cv2.IMREAD_GRAYSCALE)
    img_copy = image.copy()

    #create_test_csv(image, 'binary_mask_aerien')

    loaded_mask = np.loadtxt("test_files/binary_mask_aerien.csv", delimiter=",")
    #loaded_mask = np.loadtxt("test_files/binary_mask_chile_sinal.csv", delimiter=",")
    #loaded_mask = np.loadtxt("test_files/binary_mask_shapes.csv", delimiter=",")
    #loaded_mask = np.loadtxt("test_files/binary_mask_bateau.csv", delimiter=",")
    #loaded_mask = np.loadtxt("test_files/binary_mask_duck.csv", delimiter=",")
    #loaded_mask = np.loadtxt("test_files/binary_mask_bergamo_big.csv", delimiter=",")
    loaded_mask = loaded_mask.astype(int)
    show = loaded_mask.copy()

    image_unfilled = image * (1 - loaded_mask).astype(np.uint8)

    cv2.imshow('Image Unfilled', (image_unfilled).astype(np.uint8))

    # Initial confidence matrix
    confidence_matrix = np.ones_like(loaded_mask, dtype=np.float32) - loaded_mask

    iteration = 0

    while(len(loaded_mask[loaded_mask == 1]) != 0):
        confidence_fill_front = np.empty((0,))
        data_fill_front = np.empty((0,))

        normal_fill_front = np.empty((0, 2)) 
        gradient_fill_front = np.empty((0, 2)) 

        contour, points_contour = get_fill_front(loaded_mask)

        for pixel in points_contour:
            window_coordinates, _ = extract_window(pixel, window_size, image_unfilled, loaded_mask)
            pixel_confidence = calculate_confidence(confidence_matrix, window_coordinates)
            pixel_data = calculate_data(image_unfilled, loaded_mask, pixel, 2, 1)  ######### image or image_unfilled???

            #------------para Print Normal e gradiente------------------------#
            pixel_normal = calculate_normal(loaded_mask, pixel, 2)
            pixel_gradient = calculate_gradient(image_unfilled, loaded_mask, pixel, 3, 1)

            normal_fill_front = np.vstack([normal_fill_front, pixel_normal])
            gradient_fill_front = np.vstack([gradient_fill_front, pixel_gradient])
            #-----------------------------X-----------------------------------#

            confidence_fill_front = np.append(confidence_fill_front, pixel_confidence)
            data_fill_front = np.append(data_fill_front, pixel_data)

        #--------------------Print Normal e Gradiente-------------------------------------#
        # img_height = 100
        # img_width = 100

        # # Vetor de coordenadas (y, x) dos pixels onde queremos plotar vetores
        # coords = np.array(points_contour)

        # vectors = np.array(gradient_fill_front) /30

        # # Cria uma matriz de zeros com o tamanho da imagem para plotagem
        # image_teste_arrows = np.zeros((img_height, img_width))

        # # Plotar a imagem base como um fundo preto
        # plt.imshow(image_teste_arrows, cmap='gray', extent=(0, img_width, img_height, 0))

        # # Extrair as coordenadas e os vetores correspondentes
        # y_coords, x_coords = coords[:, 0], coords[:, 1]
        # dy, dx = vectors[:, 1], vectors[:, 0]

        # # Plotar as setas (vetores) nas posições especificadas
        # plt.quiver(x_coords, y_coords, dx, dy, angles='xy', scale_units='xy', scale=1, color='red')

        # # Ajustar e exibir a imagem
        # plt.title("Representação de Vetores na Imagem")
        # plt.xlim(0, img_width)
        # plt.ylim(img_height, 0)
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()

        #-------------------- Print dos valores de data ---------------------------#
        # # Defina as dimensões da imagem (altura, largura)
        # img_height = 100
        # img_width = 100

        # # Cria uma matriz de zeros com o tamanho da imagem
        # image_testeee = np.zeros((img_height, img_width))

        # # Atribui os valores às coordenadas especificadas
        # for (y, x), value in zip(points_contour, data_fill_front):
        #     image_testeee[y, x] = value

        # # Plot da imagem resultante
        # plt.imshow(image_testeee, cmap='gray')
        # plt.colorbar()
        # plt.title("Imagem com Valores de Data")
        # plt.show()
        # #--------------------------------X-----------------------------------------#


        # Find the patch with highest priority
        index_highest_priority = calculate_priorities(confidence_fill_front, data_fill_front)
        pixel_highest_priority = points_contour[index_highest_priority]
        patch_coordinates, patch_values = extract_window(pixel_highest_priority, window_size, image_unfilled, loaded_mask)

        # Patch matching
        fill_values, top_left_coordinate = patch_matching(image_unfilled, loaded_mask, patch_values, window_size)
        nan_indices = np.isnan(patch_values)

        # Save the current state with the chosen patch highlighted
        iteration_image = cv2.cvtColor(image_unfilled, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"{output_dir}/iteration_{iteration:03d}.png", iteration_image)

        iteration_copy = iteration_image.copy()
        top_left = top_left_coordinate
        bottom_right = (top_left[0] + window_size - 1, top_left[1] + window_size - 1)
        cv2.rectangle(iteration_copy, (top_left[1], top_left[0]), (bottom_right[1], bottom_right[0]), (0, 0, 255), 1)
        cv2.imwrite(f"{output_dir}/iteration_{iteration:03d}b.png", iteration_copy)

        for idx, (x, y) in enumerate(patch_coordinates):
            if nan_indices[idx]: 
                image_unfilled[x, y] = fill_values[idx]

                loaded_mask[x, y] = 0
                confidence_matrix[x, y] = confidence_fill_front[index_highest_priority]

        iteration += 1

    images = sorted([os.path.join(output_dir, img) for img in os.listdir(output_dir) if img.endswith(".png")])
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

    for image_path in images:
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()

    for image_path in images:
        os.remove(image_path)

    show_uint8 = (show * 255).astype(np.uint8)
    #image_unfilled = cv2.resize(image_unfilled, (200, 200))
    cv2.imshow('Updated Image', image_unfilled)
    #show_uint8 = cv2.resize(show_uint8, (200, 200))
    cv2.imshow('Initial Image', show_uint8)
    #loaded_mask = cv2.resize((loaded_mask*255).astype(np.uint8), (200, 200))
    cv2.imshow('Loaded mask', (loaded_mask*255).astype(np.uint8))
    #confidence_matrix = cv2.resize((confidence_matrix * 255).astype(np.uint8), (200, 200))
    cv2.imshow('Confidence', (confidence_matrix * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
