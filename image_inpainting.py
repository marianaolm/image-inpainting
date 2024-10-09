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


def get_matrix_target_region(image, points):
    """
    Extracts and modifies a specified target region in an image based on provided points.

    This function takes an input image and a set of points defining a polygonal region. 
    It returns a matrix of the same size as the input image where 
    the pixels inside the target region are set to white (255), while pixels outside 
    this region retain their original values.

    Parameters:
    - image (numpy.ndarray): The input image from which the target region will be extracted.
    - points (list of tuples): A list of (x, y) coordinates that define the vertices of the polygonal region.

    """
    
    contour = np.array([points], dtype=np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.fillPoly(mask, contour, 255)

    image_with_target_region = np.ones_like(image) * 255 
    image_with_target_region[mask == 0] = image[mask == 0]

    return image_with_target_region

### Main ###

if __name__ == "__main__":
    
    drawing = False
    points = [] 

    # Load the image
    image = cv2.imread('aerien1.tif')
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
        image_with_white_region = get_matrix_target_region(image, points)
        
        cv2.waitKey(0)

    cv2.destroyAllWindows()