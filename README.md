# Artistic Photo Median Stacking with Video Stabilization

This script takes an artistic approach to photo processing by median stacking the frames of a video while stabilizing them. 
It's perfect for creating unique, dream-like images from a video source, removing unwanted movement, and emphasizing stable details over time.

## Features

- **Frame Extraction**: Loads and extracts all frames from the input video.
- **Frame Registration**: Stabilizes frames by using template matching to align each frame to the first one, minimizing camera shake.
- **Median Stacking**: Combines all the stabilized frames using a median stack to create an artistic representation.
- **Outlier Rejection**: Optional pixel rejection based on standard deviation for cleaner results.

## How It Works
The script follows these main steps:
1. **Load Video Frames**: Extracts frames from the input video.
2. **Register Frames**: Aligns each frame to the reference (first) frame to mitigate unwanted motion.
3. **Median Stack**: Takes the median value across all frames to generate the final image, reducing noise and enhancing stationary elements.

## Dependencies
- OpenCV (`cv2`)
- NumPy (`numpy`)

To install the required dependencies, run:
```sh
pip install opencv-python numpy
```

## Usage
To use this script, modify the video path and crop area to suit your input video.

```python
import cv2
import numpy as np

# Load video frames
frames = load_video_frames(r'video.mp4')

# Define the crop area (x, y, width, height)
crop = (650, 150, 50, 50)

# Register (stabilize) frames
frames = register_frames(frames, crop)

# Perform median stacking on the color frames
stacked_image = median_stack(frames)

# Save the final output image
save_as_tiff(stacked_image, 'median.jpg')
```

### Parameters
- **`video_path`**: Path to the input video file.
- **`crop_rect`**: Tuple representing the area of interest for stabilization (x, y, width, height).
- **`boundary_extension`**: The number of pixels to extend around the boundary for template matching (default: 50).
- **`std_dev_threshold`**: Used for outlier rejection during median stacking (default: 2).
