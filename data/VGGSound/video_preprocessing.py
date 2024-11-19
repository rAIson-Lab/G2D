import pandas as pd
import cv2
import os

class VideoReader(object):
    def __init__(self, video_path, frame_kept_per_second=1):
        self.video_path = video_path
        self.frame_kept_per_second = frame_kept_per_second
        self.vid = cv2.VideoCapture(self.video_path)
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.video_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)

    def extract_frames(self, frame_save_path):
        count = 0
        frame_interval = int(self.fps / self.frame_kept_per_second)
        frame_id = 0  # Initialize frame_id
        while count < self.video_frames:
            ret, image = self.vid.read()
            if not ret:
                break
            if count % self.fps == 0:
                frame_id = 0
            if frame_id < frame_interval * self.frame_kept_per_second and frame_id % frame_interval == 0:
                save_name = os.path.join(frame_save_path, f'{count:05d}.jpg')
                cv2.imencode('.jpg', image)[1].tofile(save_name)
            frame_id += 1
            count += 1

class VGGSoundDataset(object):
    def __init__(self, path_to_dataset, path_to_frames, csv_path, frame_kept_per_second=1):
        self.csv_path = csv_path
        self.frame_kept_per_second = frame_kept_per_second
        self.df = pd.read_csv(self.csv_path)
        self.path_to_video = os.path.join(path_to_dataset, 'video')
        self.path_to_save_train = os.path.join(path_to_frames, f'frames/train/Image-{self.frame_kept_per_second:02d}-FPS/')
        self.path_to_save_test = os.path.join(path_to_frames, f'frames/test/Image-{self.frame_kept_per_second:02d}-FPS/')
        os.makedirs(self.path_to_save_train, exist_ok=True)
        os.makedirs(self.path_to_save_test, exist_ok=True)

    def extract_images(self):
        for index, row in self.df.iterrows():
            video_filename = f"{row['youtube_id']}_{int(row['time']):06d}.mp4"
            video_path = os.path.join(self.path_to_video, video_filename)
            frame_save_path = self.path_to_save_train if row['split'] == 'train' else self.path_to_save_test
            frame_save_path = os.path.join(frame_save_path, video_filename.split('.')[0])  # Save frames in subfolder named after the video
            os.makedirs(frame_save_path, exist_ok=True)
            try:
                video_reader = VideoReader(video_path, frame_kept_per_second=self.frame_kept_per_second)
                video_reader.extract_frames(frame_save_path)
                print(f'Processed {video_filename}')
            except Exception as e:
                print(f'Failed to process {video_filename}: {e}')

# Example usage
dataset_path = '/home/USERNAME/Multimodal-Datasets/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/' # where the original dataset is
frame_path = '/home/USERNAME/Multimodal-Datasets/VGGSound/video'
csv_file_path = '/home/USERNAME/Multi-modal-Imbalance/data/VGGSound/vggsound.csv'
vgg_dataset = VGGSoundDataset(dataset_path, frame_path, csv_file_path)
vgg_dataset.extract_images()


# import pandas as pd
# import cv2
# import os
# import pdb

# class videoReader(object):
#     def __init__(self, video_path, frame_interval=1, frame_kept_per_second=1):
#         self.video_path = video_path
#         self.frame_interval = frame_interval
#         self.frame_kept_per_second = frame_kept_per_second

#         #pdb.set_trace()
#         self.vid = cv2.VideoCapture(self.video_path)
#         self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
#         self.video_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
#         self.video_len = int(self.video_frames/self.fps)


#     def video2frame(self, frame_save_path):
#         self.frame_save_path = frame_save_path
#         success, image = self.vid.read()
#         count = 0
#         while success:
#             count +=1
#             if count % self.frame_interval == 0:
#                 save_name = '{}/frame_{}_{}.jpg'.format(self.frame_save_path, int(count/self.fps), count)  # filename_second_index
#                 cv2.imencode('.jpg', image)[1].tofile(save_name)
#             success, image = self.vid.read()


#     def video2frame_update(self, frame_save_path):
#         self.frame_save_path = frame_save_path

#         count = 0
#         frame_interval = int(self.fps/self.frame_kept_per_second)
#         while(count < self.video_frames):
#             ret, image = self.vid.read()
#             if not ret:
#                 break
#             if count % self.fps == 0:
#                 frame_id = 0
#             if frame_id<frame_interval*self.frame_kept_per_second and frame_id%frame_interval == 0:
#                 save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
#                 cv2.imencode('.jpg', image)[1].tofile(save_name)

#             frame_id += 1
#             count += 1


# class VGGSound_dataset(object):
#     def __init__(self, path_to_dataset = '/data/users/xiaokang_peng/VGGsound/', frame_interval=1, frame_kept_per_second=1):
#         self.path_to_video = os.path.join(path_to_dataset, 'test-videos/test-set/')
#         self.frame_kept_per_second = frame_kept_per_second
#         self.path_to_save = os.path.join(path_to_dataset, 'test-videos/test-set-img/', 'Image-{:02d}-FPS'.format(self.frame_kept_per_second))
#         if not os.path.exists(self.path_to_save):
#             os.mkdir(self.path_to_save)

#         videos = '/data/users/xiaokang_peng/VGGsound/test-videos/test_video_list.txt'
#         with open(videos, 'r') as f:
#             self.file_list = f.readlines()

#     def extractImage(self):

#         for i, each_video in enumerate(self.file_list):
#             if i % 100 == 0:
#                 print('*******************************************')
#                 print('Processing: {}/{}'.format(i, len(self.file_list)))
#                 print('*******************************************')
#             video_dir = os.path.join(self.path_to_video, each_video[:-1])
#             try:
#                 self.videoReader = videoReader(video_path=video_dir, frame_kept_per_second=self.frame_kept_per_second)

#                 save_dir = os.path.join(self.path_to_save, each_video[:-1])
#                 if not os.path.exists(save_dir):
#                     os.mkdir(save_dir)
#                 self.videoReader.video2frame_update(frame_save_path=save_dir)
#             except:
#                 print('Fail @ {}'.format(each_video[:-1]))


# vggsound = VGGSound_dataset()
# vggsound.extractImage()