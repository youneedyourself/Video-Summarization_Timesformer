import gradio as gr
import cv2
import os
import tempfile
import numpy as np
from utils import *
from algorithm import *
import time

def make_video(video_path, outdir='./summarized_video', algorithm='Offline (KMeans)', ratio=15, threshold_type='Average'):
    if algorithm not in ["Offline (KMeans)", "Online (Sum of Squared Difference)"]:
        algorithm = "Offline (KMeans)"
    
    if threshold_type not in ["Small", "Average", "Large"]:
        threshold_type = "Average"
    
    if threshold_type == "Small":
        threshold = 100
    elif threshold_type == "Average":
        threshold = 400
    else:
        threshold = 800


    # nen them vao cac truong hop mo hinh khac
    model, processor, device = load_model()

    # total_params = sum(param.numel() for param in model.parameters())
    # print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    if os.path.isfile(video_path):
        if video_path.endswith('txt'):
            with open(video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [video_path]
    else:
        filenames = os.listdir(video_path)
        filenames = [os.path.join(video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    for k, filename in enumerate(filenames):
        print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)
        
        raw_video = cv2.VideoCapture(filename)
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = frame_count / frame_rate
        width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #length = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
        start_time = time.time()
        filename = os.path.basename(filename)
        in_width = width
        in_height = height
        # Find the size to resize
        if "shortest_edge" in processor.size:
            height = width = processor.size["shortest_edge"]
        else:
            height = processor.size["height"]
            width = processor.size["width"] 
        resize_to = (height, width)

        # F/Fs
        clip_sample_rate = 1
        # F
        num_frames = 8
        
        original_frames = []
        frames = []
        features = []

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            output_path = tmpfile.name
        
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            # use the original frames to write the output video if you want
            #original_frames.append(raw_frame)
            raw_frame = cv2.resize(raw_frame, resize_to)
            # use the resized frames to extract features
            frames.append(raw_frame)

        # Find key frames by selecting frames with clip_sample_rate
        key_frames = frames[::clip_sample_rate] 
        #print('total of frames after sample:', len(selected_frames))

        # Remove redundant frames to make the number of frames can be divided by num_frames
        num_redudant_frames = len(key_frames) - (len(key_frames) % num_frames)

        # Final key frames
        final_key_frames = key_frames[:num_redudant_frames]
        #print('total of frames after remove redundant frames:', len(selected_frames))

        for i in range(0, len(final_key_frames), num_frames):
            if i % num_frames*50 == 0:
                print(f"Loading {i}/{len(final_key_frames)}")
        
            # Input clip to the model
            input_frames = final_key_frames[i:i+num_frames]
            # Extract features
            batch_features = extract_features(input_frames, device, model, processor)
            # Convert to numpy array to decrease the memory usage
            batch_features = np.array(batch_features.cpu().detach().numpy())
            features.extend(batch_features)

        number_of_clusters = round(len(features)*ratio/100)

        print("Total of frames: ", len(final_key_frames))
        print("Shape of each frame: ", frames[0].shape)
        print("Total of clips: ", len(features))
        print("Shape of each clip: ", features[0].shape)

        selected_frames = []
        if algorithm == "Offline (KMeans)":
            selected_frames = offline(number_of_clusters, features)
        else:
            selected_frames = online(features, threshold, ratio)

        print("Selected frame: ", selected_frames)
        
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frames[0].shape[1], frames[0].shape[0]))
        for idx in selected_frames:
            video_writer.write(frames[idx])
            # video_writer.write(original_frames[idx]) if you want to write the original frames
        
        out_duration_seconds = len(selected_frames) / frame_rate
        out_width = frames[0].shape[1]
        out_height = frames[0].shape[0]

        raw_video.release()
        video_writer.release()
        print("Completed summarizing the video (wait for a moment to load).")
        end_time = time.time()
        process_time = round(end_time - start_time, 2)
        return output_path, duration_seconds, frame_rate, f"{in_width} x {in_height}", process_time, out_duration_seconds, frame_rate, f"{out_width} x {out_height}"

css = """
#img-display-container {
    max-height: 100vh;
    }
#img-display-input {
    max-height: 80vh;
    }
#img-display-output {
    max-height: 80vh;
    }
"""

_HEADER_ = '''
<h2><b>Video summarization 🤗 Gradio Demo</b></h2><h2><a href='https://github.com/youneedyourself/Video-Summarization_Timesformer target='_blank'><b>Video Summarization Using Timesformer Modal and K-means, SSD</b></a></h2>

Code: <a href='https://github.com/youneedyourself/Video-Summarization_Timesformer' target='_blank'>GitHub</a>.
Author: Nguyen Hoai Nam.
'''

with gr.Blocks(css=css) as demo:
    gr.Markdown(_HEADER_)

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video (Required Duration > 5s)")
        with gr.Column():
            # Thêm thông tin về video
            algorithm_type = gr.Dropdown(["Offline (KMeans)", "Online (Sum of Squared Difference)"], type="value", label='Algorithm')
            ratio = gr.Slider(15, 80, label="Summarization Ratio (%) (Recommend: 15)")
            threshold = gr.Dropdown(["Small", "Average", "Large"], type="value", label='Difference Threshold for Online Algorithm (Recommend: Average)')
            submit = gr.Button("Summarize")
    with gr.Row():      
        processed_video = gr.Video(label="Summarized Video")
        with gr.Column():
            input_video_duration = gr.Text(label="Input Video Duration (s)")
            input_video_FPS = gr.Text(label="Input Video FPS")
            input_video_resolution = gr.Text(label="Input Video Resolution")
        
        time_process = gr.Text(label="Time Process (s)")

        with gr.Column():
            output_video_duration = gr.Text(label="Output Video Duration (s)")
            output_video_FPS = gr.Text(label="Output Video FPS")
            output_video_resolution = gr.Text(label="Output Video Resolution")

    def on_submit(uploaded_video, algorithm_type, ratio, threshold):
        print("Ratio:", ratio)
        print("Algorithm:", algorithm_type)
        print("Threshold:", threshold)
        # Process the video and get the path of the output video
        output_video_path = make_video(uploaded_video, algorithm=algorithm_type, ratio=ratio, threshold_type=threshold)
        return output_video_path

    submit.click(on_submit, inputs=[input_video, algorithm_type, ratio, threshold], outputs=[processed_video, input_video_duration, input_video_FPS, input_video_resolution, time_process, output_video_duration, output_video_FPS, output_video_resolution])

if __name__ == '__main__':
    demo.queue().launch(share=True)