import csv
import os
import librosa
import numpy as np
import torch
import config
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CremadDataset(Dataset):

    def __init__(self, mode='train'):
        self.mode = mode
        self.data_root = ''
        self.class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

        self.visual_path = config.visual_path
        self.audio_path = config.audio_path

        self.train_csv = config.train_path
        self.test_csv = config.test_path
        csv_file = self.train_csv if mode == 'train' else self.test_csv

        self.image = []
        self.audio = []
        self.label = []

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_path, item[0])
                
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(self.class_dict[item[1]])
                else:
                    continue

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]
        data = []

        # Load and process audio
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050*3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        #mean = np.mean(spectrogram)
        #std = np.std(spectrogram)
        #spectrogram = np.divide(spectrogram - mean, std + 1e-9)
        audio_tensor = torch.tensor(spectrogram, dtype=torch.float32)
        data.append(audio_tensor)

        # Load and process visual
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224) if self.mode == 'train' else transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip() if self.mode == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load and process video
        image_samples = os.listdir(self.image[idx])
        image_samples = sorted(image_samples)
        images = torch.zeros((config.fps, 3, 224, 224))  
        np.random.seed(999)
        select_index = np.random.choice(len(image_samples), size=config.fps, replace=False)   # Set size to the no. of frames extracted per second (fps)

        # with select_index, an random image will be loaded from all the images in the folder, this will cause inconsistency while evaluating the model
        img = Image.open(os.path.join(self.image[idx], image_samples[int(select_index)])).convert('RGB')
        # without select_index, alwayus the first image in the folder will be loaded
        # img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
        images[0] = transform(img)

        # print(f"Images shape: {images.shape}")  # (1, 3, 224, 224) where 1 is the number of frames, 3 is the number of channels, 224 is the height and 224 is the width
        video_tensor = torch.permute(images, (1, 0, 2, 3))  
        # print(f"Video tensor shape: {video_tensor.shape}") # (3, 1, 224, 224)
        data.append(video_tensor)

        return (*data, label)

