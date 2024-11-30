import cv2
import numpy as np

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # Keep color information
    cap.release()
    return frames

def register_frames(frames, crop_rect, boundary_extension=50):
    """
    Registers frames based on template matching.

    Args:
    - frames (list of np.array): List of frames in color.
    - crop_rect (tuple): Rectangle (x, y, width, height) defining the crop area in the first frame.
    - boundary_extension (int): Number of pixels to extend the boundary for template matching.

    Returns:
    - list of np.array: List of registered frames.
    """
    # Extract the template from the first frame (reference frame)
    x, y, w, h = crop_rect
    template = frames[0][y:y+h, x:x+w]
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    registered_frames = [frames[0]]  # Start with the first frame already included
    
    for frame in frames[1:]:
        # Define the search area in the new frame
        search_area = frame[
            max(y - boundary_extension, 0) : min(y + h + boundary_extension, frame.shape[0]),
            max(x - boundary_extension, 0) : min(x + w + boundary_extension, frame.shape[1])
        ]
        search_area_gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
        
        # Template matching
        res = cv2.matchTemplate(search_area_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        
        # Calculate the displacement
        top_left = (max_loc[0] + max(x - boundary_extension, 0), max_loc[1] + max(y - boundary_extension, 0))
        translation = (top_left[0] - x, top_left[1] - y)
        print(translation)
        # Apply the translation to align the frame
        translation_matrix = np.float32([[1, 0, -translation[0]], [0, 1, -translation[1]]])
        aligned_frame = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))
        
        registered_frames.append(aligned_frame)
    
    return registered_frames

def median_stack_with_rejection(frames, std_dev_threshold=2):
    """
    Performs median stacking on a list of frames, rejecting pixels far from the mean.

    Args:
    - frames (list of np.array): List of image frames.
    - std_dev_threshold (float): Number of standard deviations to use as the threshold for outlier rejection.

    Returns:
    - np.array: The stacked image.
    """
    # Convert the list of frames to a numpy array
    array_frames = np.array(frames)

    # Calculate the mean and standard deviation along the stack
    mean = np.mean(array_frames, axis=0)
    std_dev = np.std(array_frames, axis=0)

    # Identify pixels to keep: those within the defined number of standard deviations from the mean
    lower_bound = mean - std_dev_threshold * std_dev
    upper_bound = mean + std_dev_threshold * std_dev

    # Create a mask for valid pixels
    valid_mask = (array_frames >= lower_bound) & (array_frames <= upper_bound)

    # Apply the mask and replace outliers with NaN for median calculation
    filtered_frames = np.where(valid_mask, array_frames, np.nan)

    # Calculate the median on the filtered stack
    stack = np.nanmedian(filtered_frames, axis=0).astype(np.uint8)

    return stack


def median_stack(frames):
    array_frames = np.array(frames)
    stack = np.median(array_frames, axis=0) #.astype(np.uint32)
    return stack

def save_as_tiff(image, file_path):
    cv2.imwrite(file_path, image)


crop=(650,150,50,50)

# Replace 'path_to_video.mp4' with the path to your video file
frames = load_video_frames(r'video.mp4')
frames=register_frames(frames, crop)

# Perform median stacking on the color frames
stacked_image = median_stack(frames)
#stacked_image =median_stack_with_rejection(frames)
# Replace 'output_image.tiff' with the desired TIFF file path
save_as_tiff(stacked_image, 'median.jpg')
