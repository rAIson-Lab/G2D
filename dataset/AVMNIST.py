import torch
import config
import os
import numpy as np
from torch.utils.data import Dataset

class AVMNIST(Dataset):
    def __init__(self,mode = 'train') -> None:
        super(AVMNIST,self).__init__()
        image_data_path = os.path.join(config.data_root,'image')
        audio_data_path = os.path.join(config.data_root,'audio')
        
        if mode == 'train':
            self.train_image = np.load(os.path.join(image_data_path,'train_data.npy'))
            self.train_audio = np.load(os.path.join(audio_data_path,'train_data.npy'))
            self.train_label = np.load(os.path.join(config.data_root,'train_labels.npy'))
            # Keep val/dep set
            # self.image = self.train_image[:int(self.train_image.shape[0]*0.9)]
            # self.audio = self.train_audio[:int(self.train_audio.shape[0]*0.9)]
            # self.label = self.train_label[:int(self.train_label.shape[0]*0.9)]
            # No val/dep set
            self.image = self.train_image
            self.audio = self.train_audio
            self.label = self.train_label
            
        elif mode == 'test':
            self.test_image = np.load(os.path.join(image_data_path,'test_data.npy'))
            self.test_audio = np.load(os.path.join(audio_data_path,'test_data.npy'))
            self.test_label = np.load(os.path.join(config.data_root,'test_labels.npy'))
            # Keep val/dep set
            # self.image = self.test_image[:int(self.test_image.shape[0]*0.9)]
            # self.audio = self.test_audio[:int(self.test_audio.shape[0]*0.9)]
            # self.label = self.test_label[:int(self.test_label.shape[0]*0.9)]
            # No val/dep set
            self.image = self.test_image
            self.audio = self.test_audio
            self.label = self.test_label
        # elif mode == 'dep':
        #     self.train_image = np.load(os.path.join(image_data_path,'train_data.npy'))
        #     self.train_audio = np.load(os.path.join(audio_data_path,'train_data.npy'))
        #     self.train_label = np.load(os.path.join(config.data_root,'train_labels.npy'))
            
        #     self.test_image = np.load(os.path.join(image_data_path,'test_data.npy'))
        #     self.test_audio = np.load(os.path.join(audio_data_path,'test_data.npy'))
        #     self.test_label = np.load(os.path.join(config.data_root,'test_labels.npy'))

        #     self.image = np.concatenate([self.train_image[int(self.train_image.shape[0]*0.9):],self.test_image[int(self.test_image.shape[0]*0.9):]])
        #     self.audio = np.concatenate([self.train_audio[int(self.train_audio.shape[0]*0.9):],self.test_audio[int(self.test_audio.shape[0]*0.9):]])
        #     self.label = np.concatenate([self.train_label[int(self.train_label.shape[0]*0.9):],self.test_label[int(self.test_label.shape[0]*0.9):]])
        self.length = len(self.image)
        
    def __getitem__(self, index):
        image = self.image[index]
        audio = self.audio[index]
        label = self.label[index]
        
        # Normalize image
        image = image / 255.0
        audio = audio / 255.0
        
        image = image.reshape(28,28)
        image = np.expand_dims(image,0)
        audio = np.expand_dims(audio,0)
        
        image = torch.from_numpy(image)
        audio = torch.from_numpy(audio)
        return audio,image,label
    
    def __len__(self):
        return self.length