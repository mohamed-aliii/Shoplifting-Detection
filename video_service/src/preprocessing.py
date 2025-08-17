import torch
import torchvision.io as io

'''def preprocess_video(video_path, num_frames=16, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Preprocess a video to match the model's expected input format.
    
    Steps:
    1. Read video frames using torchvision.io.read_video
    2. Select/pad frames to num_frames
    3. Normalize pixel values to [0, 1]
    4. Permute dimensions to (1, C, T, H, W)
    5. Move tensor to the specified device
    """
    # Read video
    try:
        video, _, _ = io.read_video(video_path, pts_unit='sec')
    except Exception as e:
        raise RuntimeError(f"Error reading video {video_path}: {e}")

    total_frames = video.shape[0]
    if total_frames == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    # Select frames
    if total_frames < num_frames:
        frame_indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        # Uniform sampling across the video
        frame_indices = torch.linspace(0, total_frames - 1, num_frames).long()

    selected_frames = video[frame_indices]  # Shape: (num_frames, H, W, C)

    # Normalize to [0, 1]
    selected_frames = selected_frames.float() / 255.0

    # Permute to (num_frames, C, H, W)
    selected_frames = selected_frames.permute(0, 3, 1, 2)

    # Add batch dimension and match model input shape: (1, C, T, H, W)
    input_tensor = selected_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)

    # Move to device
    input_tensor = input_tensor.to(device)

    return input_tensor'''

import os
import torch
import torchvision.io as io
import torchvision.transforms as T

def preprocess_video(video_path, num_frames=16, target_size=112):
    """
    Preprocess a single video for inference by extracting and preparing frames.
    This matches the training preprocessing: extracts 16 frames (padding with duplicates if needed),
    resizes to 112x112, normalizes to [0,1], and returns a tensor of shape (1, 16, 3, 112, 112).
    
    Args:
        video_path (str): Path to the input video file.
        num_frames (int): Number of frames to extract (default: 16).
        target_size (tuple): Target frame size (height, width) (default: (112, 112)).
    
    Returns:
        torch.Tensor: Preprocessed video tensor ready for model input (1, T, C, H, W).
    
    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video cannot be opened.
        ValueError: If no frames are found in the video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        video, _, info = io.read_video(video_path, pts_unit='sec')
    except Exception as e:
        raise RuntimeError(f"Cannot open video: {video_path} - {e}")

    total_frames = video.shape[0]
    if total_frames == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    # Select frame indices (evenly spaced, pad with last frame if fewer than num_frames)
    if total_frames < num_frames:
        frame_indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
        frame_indices = torch.tensor(frame_indices, dtype=torch.long)
    else:
        frame_indices = torch.linspace(0, total_frames - 1, steps=num_frames).long()

    # Select frames: (num_frames, H_orig, W_orig, 3) uint8 RGB
    selected_frames = video[frame_indices]

    # Normalize to [0,1] float32
    selected_frames = selected_frames.float() / 255.0

    # Permute to (num_frames, 3, H_orig, W_orig)
    selected_frames = selected_frames.permute(0, 3, 1, 2)

    # Resize each frame to target_size
    resize_transform = T.Resize(112)
    selected_frames = torch.stack([resize_transform(frame) for frame in selected_frames])

    # Add batch dimension: (1, num_frames, 3, H, W)
    selected_frames = selected_frames.unsqueeze(0)

    return selected_frames
