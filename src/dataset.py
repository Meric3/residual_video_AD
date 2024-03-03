from torch.utils.data import Dataset
import pdb
import os 
import numpy as np
import scipy.io
import cv2
import torch

from convlstm import *



class Video_dataset(Dataset):
    def __init__(self, dataset='avenue', window_size=16,
        resize_height=224,resize_width=224):
        
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.window_size = window_size

        self.train_video_path = './data/Avenue Dataset/training_frames/'

        self.train_videos_list = np.sort(os.listdir(self.train_video_path))

        self.folder_num = len(self.train_videos_list)

        # cal len
        # 각각의 폴더 당 프레임 개수를 알고있어야 나중에 계산할수 있다.
        self.folder_frame_len = np.zeros(self.folder_num, dtype=np.int16)
        self.folder_frame = []
        for i in range(len(self.train_videos_list)):
            self.folder_frame_len[i] = len(os.listdir(os.path.join(\
            self.train_video_path, self.train_videos_list[i])))

            self.folder_frame.extend(os.listdir(os.path.join(\
            self.train_video_path, self.train_videos_list[i])))

        self.folder_frame = np.array(self.folder_frame)

            


    def __len__(self):
        return np.sum(self.folder_frame_len) 

    def __getitem__(self, index):
        resized_frames = []
        select_num = 0
        for i in range(len(self.folder_frame_len)):
            select_num += self.folder_frame_len[i]
            if select_num > index:
                break

        if index + self.window_size > select_num:
            index = index - self.window_size


        prv = scipy.misc.imread(os.path.join(self.train_video_path, self.train_videos_list[i],\
        self.folder_frame[index]), mode='L')
        prv = scipy.misc.imresize(prv, (self.resize_width, self.resize_height))

        y = scipy.misc.imread(os.path.join(self.train_video_path, self.train_videos_list[i],\
        self.folder_frame[index + self.window_size]), mode='L')
        y = scipy.misc.imresize(y, (self.resize_width, self.resize_height))
        y = np.expand_dims(y, axis=-1)/255.


        for next_idx in range(self.window_size):
            nxt = scipy.misc.imread(os.path.join(self.train_video_path, self.train_videos_list[i],\
            self.folder_frame[index + next_idx + 1]), mode='L')
            nxt = scipy.misc.imresize(nxt, (self.resize_width, self.resize_height))

            flow = cv2.calcOpticalFlowFarneback(prv, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

            hsv = np.zeros((self.resize_width, self.resize_height,3))
            hsv[...,1] = 255.
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv = np.uint8(hsv)

            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            rgb = rgb/255.

            nxt = np.expand_dims(nxt, axis=-1)
            frame = np.concatenate([nxt/255., rgb], axis=-1)
            resized_frames.append(frame)

        resized_frames = np.stack(resized_frames, axis=0)

        return torch.from_numpy(resized_frames), torch.from_numpy(y)







if __name__ == "__main__":
    from torch.utils.data import DataLoader 
    train_data = Video_dataset(dataset='avenue', window_size=10)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)	

    model = ConvLSTM(input_size=(224,224),
    input_dim=4,
    hidden_dim=[32,32,64],
    kernel_size=(3, 3),
    num_layers=3,
    batch_first=True,
    bias=True,
    return_all_layers=False)

    model.to('cuda')

    for i, sample in enumerate(train_loader):
        tp = model(sample[0].to('cuda').type(torch.cuda.FloatTensor))
        print(i)
        pdb.set_trace()





