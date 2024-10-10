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




### Main ###

if __name__ == "__main__":
    
    drawing = False
    points = [] 

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

    # After point selection, obtain the target region matrices
    if len(points) >= 3: 
        binary_mask, contour = get_matrix_mask(image, points)
        
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Initial confidence matrix 
    confidence_matrix = np.ones_like(binary_mask, dtype=np.uint8) - binary_mask
