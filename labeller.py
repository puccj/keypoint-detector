import cv2
import numpy as np
import os
import random
import argparse
from tqdm import tqdm

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['mousePosition'] = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        param['mousePosition'] = (-2, -2)

def select_points(image):
    cv2.namedWindow('Select Points')
    param = {'mousePosition': None}
    cv2.setMouseCallback('Select Points', on_mouse, param)
    
    points = []

    key = -1
    while key != ord(' '):  # Exit on 'Space' key
        display_image = image.copy()
        for point in points:
            cv2.drawMarker(display_image, point, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 2)

        cv2.imshow('Select Points', display_image)

        key = cv2.waitKey(20) & 0xFF


        if param['mousePosition'] is not None:
            if param['mousePosition'] == (-2, -2): # Remove last point
                if points:
                    points.pop()
            else:
                points.append(param['mousePosition'])
            param['mousePosition'] = None

        
        if key == ord('q') or key == 27:  # Exit on 'q' or 'ESC' key
            cv2.destroyWindow('Select Points')
            return []

    cv2.destroyAllWindows()
    return points

def label_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(input_dir, file))

    if not image_files:
        print("No image files found in the specified directory.")
        return
    
    print(f"Found {len(image_files)} image(s) in the directory.")

    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read {image_path}")
            continue

        points = select_points(image)

        if not points:
            print(f"No points selected for {image_path}. Skipping.")
            continue

        output_filename = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w') as f:
            for point in points:
                f.write(f"{point[0]} {point[1]}\n")

        print(f"Saved labels for {image_path} to {output_path}")

def load_fisheye_params(path):
    """
    Load fisheye parameters from a file and return the camera intrinsic matrix (K) and distortion coefficients (D).

    Parameters
    ----------
    path : str
        Path to the file containing the fisheye parameters. The file should contain lines in the format 'key=value' for fx, fy, cx, cy, k1, k2, p1, p2.

    Returns
    -------
    K : np.array (3x3)
        Camera matrix (intrinsic parameters)
    D : np.array (1x4)
        Distortion coefficients (radial and tangential)

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    ValueError
        If no parameters are found in the file.
    """
    parameters = {}
    
    with open(path, 'r') as file:
        for line in file:
            # Split the line by '=' to separate key and value
            key, value = map(str.strip, line.strip().split('='))
            parameters[key] = float(value)  # Convert value to float

    if not parameters:
        raise ValueError("Error: No parameters found in the file.")

    fx = parameters['fx']
    fy = parameters['fy']
    cx = parameters['cx']
    cy = parameters['cy']
    k1 = parameters['k1']
    k2 = parameters['k2']
    p1 = parameters['p1']
    p2 = parameters['p2']

    mtx = np.array([[fx, 0., cx],
                    [0., fy, cy],
                    [0., 0., 1.]])
    dist = np.array([[k1, k2, p1, p2]])

    return mtx, dist

def extract_screenshots(video_dir, output_dir, num_screenshots=200, fisheye_matrix=None):
    """
    Extract screenshots from videos in a directory and save them to an output directory.
    
    Parameters
    ----------
    video_dir : str
        Path to the directory containing the videos.
    output_dir : str
        Path to the output directory where screenshots are saved.
    num_screenshots : int
        Number of screenshots to extract from each video.
    fisheye_matrix : tuple of np.array
        Fisheye camera matrix and distortion coefficients (K, D) if fisheye correction is needed.

    Notes
    -----
    It is assumed that all the videos have the same resolution and fisheye parameters: 
    the same fisheye correction maps are applied to all videos. 
    If the videos have different resolutions, the fisheye correction will not be accurate for all of them.

    Returns
    -------
    screenshot_count : int
        Total number of screenshots extracted and saved.
    """

    os.makedirs(output_dir, exist_ok=True)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = []
    for file in os.listdir(video_dir):
        if file.lower().endswith(video_extensions):
            video_files.append(os.path.join(video_dir, file))

    if not video_files:
        print("No video files found in the specified directory.")
        return 0
    
    print(f"Found {len(video_files)} video(s) in the directory.")

    # In case of multiple runs, avoid selecting the same frame again
    existing_frames = set()
    for file in os.listdir(output_dir):
        if file.lower().endswith('.jpg'):
            existing_frames.add(file)

    # Calculate fisheye correction maps if parameters are provided
    if fisheye_matrix is not None:
        print("Fisheye parameters loaded. Preparing correction maps...")    
        # Get the size of the first video frame for fisheye correction
        first_video_path = video_files[0]
        cap = cv2.VideoCapture(first_video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open {first_video_path} for fisheye correction.")
            return 0
        ret, original = cap.read()
        cap.release()
        if not ret:
            print(f"Warning: Could not read frame from {first_video_path} for fisheye correction.")
            return 0
        size = (original.shape[1], original.shape[0])

        K, D = fisheye_matrix
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, size, cv2.CV_16SC2)

    frames_per_video = num_screenshots // len(video_files)

    saved_number = 0
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}")
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count == 0:
            print(f"Warning: No frames found in {video_path}")
            cap.release()
            continue
        
        frame_indices = random.sample(range(frame_count), min(frames_per_video, frame_count))

        for frame_idx in tqdm(frame_indices, desc=f"Processing {os.path.basename(video_path)}"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_idx} from {video_path}")
                continue
        
            # Create filename with video name and frame index
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{video_name}_frame_{frame_idx}.jpg"
            output_path = os.path.join(output_dir, output_filename)

            # Skip if the frame already exists (from previous runs)
            if output_filename in existing_frames:
                print(f"Skipping {output_filename} as it already exists.")
                continue

            # Apply fisheye correction if parameters are provided
            if fisheye_matrix is not None:
                frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            cv2.imwrite(output_path, frame)
            saved_number += 1
        
        cap.release()

    print(f"Successfully saved {saved_number} screenshots to {output_dir}")
    return saved_number

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract screenshots from videos with optional fisheye correction.")
    parser.add_argument("input_dir", type=str, 
                        help="Path to the input directory. It should contain the extracted images "
                             "or, if --extract is used, the videos from which to extract screenshots.")
    parser.add_argument("output_dir", type=str, 
                        help="Path to the output directory where labels (or screenshots if --extract is used) will be saved.")
    parser.add_argument("--extract", "-e", action="store_true", help="Flag to extract screenshots from videos.")
    parser.add_argument("--num", "-n", type=int, default=200, help="Number of screenshots to extract from each video (only used if --extract is set).")
    parser.add_argument("--fisheye", "-f", type=str, default="fishcam-fisheye.txt", 
                        help="Path to the file containing fisheye parameters for correction (only used if --extract is set). "
                        "The file should contain lines in the format 'key=value' for fx, fy, cx, cy, k1, k2, p1, p2.")
    args = parser.parse_args()


    if args.extract:
        if args.fisheye is not None:
            try:
                fisheye_matrix = load_fisheye_params(args.fisheye)
            except (FileNotFoundError, ValueError) as e:
                print(e)
                fisheye_matrix = None
        total_screenshots = extract_screenshots(args.input_dir, args.output_dir, args.num, fisheye_matrix)

        print(f"Total screenshots extracted: {total_screenshots}")
    
    else:
        label_images(args.input_dir, args.output_dir)