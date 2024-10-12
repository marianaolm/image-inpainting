import cv2
import numpy as np

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
            cv2.circle(img_copy, point, 5, (0, 0, 255), -1)

            if i > 0:
                cv2.line(img_copy, points[i - 1], points[i], (0, 255, 0), 2)

        cv2.imshow('Image', img_copy)


def get_matrix_mask(image, points):
    """
    Extracts a binary mask and the contour for a specified polygonal region.

    This function takes an input image and a set of points defining a polygonal region. 
    It returns:
    - A binary mask where 1 represents the pixels inside the polygonal region 
      and 0 represents the pixels outside.
    - The contour (polygon) used to generate the mask.

    Parameters:
    - image (numpy.ndarray): The input image from which the target region will be defined.
    - points (list of tuples): A list of (x, y) coordinates that define the vertices of the polygonal region.

    Returns:
    - binary_mask (numpy.ndarray): A binary mask where 1 represents the target region and 0 represents the rest.
    - contour (numpy.ndarray): The contour used to create the mask.
    """
    
    # Create the polygon from the provided points
    contour = np.array([points], dtype=np.int32)
    
    # Create the mask of the same size as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Fill the polygon in the mask with 255
    cv2.fillPoly(mask, contour, 255)

    # Create a binary mask (1 for region, 0 for outside)
    binary_mask = np.zeros_like(mask, dtype=np.uint8)
    binary_mask[mask == 255] = 1

    return binary_mask, contour


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
    
    # Create a copy of the image where the target region will be highlighted
    image_with_target_region = np.ones_like(image) * 255 
    image_with_target_region[binary_mask == 0] = image[binary_mask == 0]

    return image_with_target_region


def extract_window(pixel, window_size, matrix):
    """
    Extracts the coordinates of the pixels in a window around a given pixel
    and retrieves the corresponding values from the given matrix.
    
    Parameters:
    pixel (tuple): A tuple (x, y) representing the coordinates of the central pixel.
    window_size (int): The size of the window (must be an odd number for a symmetric patch).
    matrix (np.ndarray): A 2D array representing the matrix from which to extract values.
    
    Returns:
    tuple: A tuple containing two elements:
        - list: A list of tuples representing the coordinates of the pixels in the window around the given pixel.
        - list: A list of values from the matrix corresponding to the extracted window coordinates.
    """
    
    x, y = pixel
    half_window = window_size // 2
    
    window_coordinates = []
    window_values = []

    for i in range(-half_window, half_window + 1):
        for j in range(-half_window, half_window + 1):
            window_x = x + i
            window_y = y + j
            window_coordinates.append((window_x, window_y))
            
            if 0 <= window_x < matrix.shape[0] and 0 <= window_y < matrix.shape[1]:
                window_values.append(matrix[window_x, window_y])
            else:
                window_values.append(None)

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


def calculate_data(image, mask_target_region, pixel): # Pedro
    """
    Calculates the data term associated with a pixel.

    D = norm of the scalar produt between the orthogonal of the gradient of the image in the point p 
    and the normal of the target region border in the point p, all divided by alpha (= 255 for a typical
    grey-level image)?

    OKAY?
    """
    return data_value


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


### Main ###

if __name__ == "__main__":
    
    drawing = False
    points = []

    confidence_fill_front = np.empty((0,))
    data_fill_front = np.empty((0,))

    window_size = 9 # n window size (nxn)

    # Load the image
    image = cv2.imread('image-inpainting/images/aerien1.tif')
    img_copy = image.copy()

    # Display the image and set up the mouse callback
    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', select_points)

    # Wait for the user to finish selecting points
    while True:
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
            break

    # After point selection, obtain the binary mask and the contour
    if len(points) >= 3: 
        binary_mask, contour = get_matrix_mask(image, points)
        
        # Press a key to close the window
        cv2.waitKey(0)

    cv2.destroyAllWindows()


    # Imagem com a binary matrix (valores 0 na partre unfilled e valores reais fora dela)
    image_unfilled = image * (1 - binary_mask)

    # Initial confidence matrix 
    confidence_matrix = np.ones_like(binary_mask, dtype=np.uint8) - binary_mask

    # Loop que calcula a C(p) e D(p) de cada pixel do fill front (contour) gerando vetores e depois seta o vetor prioridade que segue o index do vetor contour
    for pixel in contour:
        pixel_confidence = calculate_confidence(confidence_matrix, pixel, window_size)
        pixel_data = calculate_data(image, mask_target_region, pixel)

        confidence_fill_front = np.append(confidence_fill_front, pixel_confidence)
        data_fill_front = np.append(data_fill_front, pixel_data)
    
    # Find the patch with highest priority
    index_highest_priority = calculate_priorities(confidence_fill_front, data_fill_front)
    pixel_highest_priority = contour[index_highest_priority]
    patch_coordinates, patch_values = extract_window(pixel_highest_priority, window_size, image_unfilled)

    # Patch matching


    # Após o preenchimento do pixel precisa mapear os pontos e retornar para a lista points para poder gerar a nova binary mask e contour
