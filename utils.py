from transformers import TimesformerModel, VideoMAEImageProcessor
import torch
import cv2
import numpy as np
from torchvision.transforms import Lambda
from pytorchvideo.transforms import (
    Normalize,
)
from torchvision.transforms import (
    Lambda,
)
import os
from os.path import isfile, join, basename

def extract_features(frames, device, model, image_processor):
    # Convert frames to tensor
    frames_tensor = torch.stack([torch.from_numpy(frame) for frame in frames])
    # Change the order of the tensor to (num_frames, channel, height, width)
    frames_tensor = frames_tensor.permute(3, 0, 1, 2).to(device)

    # Get the mean and std of the image processor
    mean = image_processor.image_mean
    std = image_processor.image_std

    # Normalize frames
    frames_tensor = Lambda(lambda x: x / 255.0)(frames_tensor)
    frames_tensor = Normalize(mean, std)(frames_tensor)

    # Change the order of the tensor to (num_frames, channel, height, width) and add a batch dimension
    frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)

    # Load the model to the device
    model.to(device)
    model.eval()
    outputs = model(frames_tensor)

    # Get the output after the Transformer Encoder (MLP head)
    final_output = outputs[0][:, 0]

    return final_output
    
def to_video(selected_frames, frames, output_path, video_fps):
    
    print("MP4 Format.")
    # Write the selected frames to a video
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frames[0].shape[1], frames[0].shape[0]))

    # selected_frames is a list of indices of frames
    for idx in selected_frames:
        video_writer.write(frames[idx])
    
    video_writer.release()
    print("Completed summarizing the video (wait for a moment to load).")

def load_model():
    try:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = TimesformerModel.from_pretrained(f"facebook/timesformer-base-finetuned-k600")
        processor=VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        return model, processor, DEVICE
    
    except Exception as e:
        print(e)

def sum_of_squared_difference(vector1, vector2):
    squared_diff = np.square(vector1 - vector2)
    sum_squared_diff = np.sum(squared_diff)
    return sum_squared_diff
