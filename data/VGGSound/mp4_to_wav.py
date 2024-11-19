import os
import pandas as pd

# Define the base directory for videos and audios
base_video_dir = '/home/USERNAME/Multimodal-Datasets/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video'  # Update with the correct path to the directory containing all video files
base_audio_dir = '/home/USERNAME/Multimodal-Datasets/VGGSound/audio'  # Update with the correct path to the directory where WAV files should be stored

# Load the CSV file
df = pd.read_csv('/home/USERNAME/Multi-modal-Imbalance/data/VGGSound/vggsound.csv')  # Update with the correct path to your CSV file

# Function to process videos and save them in split specific folders
def process_videos(row):
    youtube_id = row['youtube_id']
    time_stamp = format(row['time'], '06')  # Assuming time needs to be zero-padded to 6 digits
    split_type = row['split']
    file_name = f"{youtube_id}_{time_stamp}.mp4"
    mp4_filename = os.path.join(base_video_dir, file_name)
    split_audio_dir = os.path.join(base_audio_dir, split_type)  # Directory path based on split

    # Ensure the split specific directory exists
    if not os.path.exists(split_audio_dir):
        os.makedirs(split_audio_dir)
    
    wav_filename = os.path.join(split_audio_dir, f"{youtube_id}_{time_stamp}.wav")
    
    if not os.path.exists(wav_filename):
        os.system(f'ffmpeg -i "{mp4_filename}" -acodec pcm_s16le -ar 16000 "{wav_filename}"')
        print(f'Converted {mp4_filename} to {wav_filename}')

# Apply the function to each row in the dataframe
df.apply(process_videos, axis=1)