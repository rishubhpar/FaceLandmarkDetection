import numpy as np
import os
import cv2
from PIL import Image, ImageOps
import torchvision
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from pts_loader import load
#%%

class faceDB300NW():

    def __init__(self, dataset_dir, split, tforms):
        self.dataset_dir = dataset_dir
        self.split = split
        self.tforms = tforms
        self.image_paths = []
        
        if(self.split == 'train'):
            self.dataset_list = ['lfpw/trainset', 'helen/trainset', 'afw'] # Total 3148 images
            
        elif (self.split == 'val' or self.split == 'test_common'):
            self.dataset_list = ['lfpw/testset', 'helen/testset']
            
        elif (self.split == 'test_challenging'):
            self.dataset_list = ['small_test_set']
        
        elif (self.split == 'full_test_set'):
            self.dataset_list = ['lfpw/testset', 'helen/testset', 'ibug']
           
        elif (self.split == 'nikhar'):
            self.dataset_list = ['small_test_set']
            
        # Traversing over all the subdataset and appending the image path to the global list
        
        for dataset in self.dataset_list:
            dataset_path = os.path.join(self.dataset_dir, dataset)
            print('dataset paths : ', dataset_path)
            for file_ in os.listdir(dataset_path):
                if(file_[-4: ] == '.jpg' or file_[-4:] == '.png'):
                    image_prefix_path = os.path.join(dataset_path, file_)
                    self.image_paths.append(image_prefix_path)
                    
        print("Dataset Type: ", self.split, " Number of images : ", len(self.image_paths))
        
    
    def __len__(self):
        return (len(self.image_paths))
        
    def display_landmarks(self, image, heatmaps):
        canvas = image.copy()
        for i in range(0, heatmaps.shape[0]):
            for j in range(0, heatmaps.shape[1]):
                if(heatmaps[i,j,0] == 1):
                    canvas[i-2: i+2, j-2: j+2] = [255,0,0]
                    
        cv2.imshow('landmarks', canvas)
        cv2.waitKey(2000)
        
    
    # To generate heatmaps from a set of landmarks points, heatmap will be a volume where there will be a single heatmap for each of the landmark point
    def get_heatmaps(self, landmarks, shape):
        heat_map = np.zeros((shape[0], shape[1], 5))
        for pt in landmarks:
            x,y = int(pt[0]) , int(pt[1])
            heat_map[y,x,:] = 1
        return heat_map
        
    def transformHelper(self, image, image_heatmaps, flip_prob, scale_ratio, rotate_angle):
        heatmaps = None
        newDim = (int(scale_ratio * 64), int(scale_ratio * 64))
        image = cv2.resize(image, newDim, interpolation = cv2.INTER_CUBIC)
        
        if(flip_prob > 0.5):
            image = np.flip(image, 1)
            
        rows_rot, cols_rot, w = image.shape
        M = cv2.getRotataionMatrix2D((cols_rot/2, rows_rot/2), rotate_angle, 1)
        image = cv2.warpAffine(image, M, (cols_rot, rows_rot))
        
        rows_crop, cols_crop, w = image.shape
        
        if(rows_crop < 64 or cols_crop < 64):
            crop_flag = 0
            image = cv2.resize(image, (64,64), interpolation = cv2.INTER_CUBIC)
        else:
            crop_flag = 1
            x = rows_crop - 64
            y = cols_crop - 64
            xRand = random.randint(0,x)
            yRand = random.randint(0,y)
            image = image[yRand:yRand+64, xRand:xRand+64]
            
        for i in range(w):
            tmp = image_heatmaps[:,:,i].astype('unit8')
            res = cv2.resize(tmp, newDim, interpolation = cv2.INTER_CUBIC)
            if(flip_prob > 0.5):
                res = np.flip(res, 1)
                
            res = cv2.warpAffine(res, M, (cols_rot, rows_rot))
            
            if(crop_flag == 0)
                res = cv2.resize(res, (64,64), interpolation = cv2.INTER_CUBIC)
            else:
                res = res[yRand:yRand+64, xRand:xRand+64]
                
            if heatmaps is None:
                heatmaps = res
            else:
                heatmaps = np.concatenate(heatmaps, res), axis = 2)
                
        return image, heatmaps
        
        
    def transforms(self, image, image_heatmaps):
        
        flipRand = random.random()
        scaleRand = random.uniform(0.9,1.1)
        rotateRand = random.randint(0,30)
        
        print(flipRand, scaleRand, rotateRand)
        
        image, heatmaps = self.transformHelper(image, image_heatmaps, flipRand, scaleRand, rotateRand)
        return image, image_heatmaps
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        landmark_path = image_path[:-4] + '.pts'
        
        print(image_path, landmark_path)
        
        image = cv2.imread(image_path)
        landmark_path = load(landmark_path)
        print('num landmarks : ', len(landmarks))
        
        image_heatmaps = self.get_heatmaps(landmarks, image.shape)
        
        #self.display_landmarks(image, image_heatmaps)
        
        print('shapes : ', image,shape, image_heatmaps.shape)
        
        # Custom transforms to process the image and the heatmaps together
        
        image, image_heatmaps = self.transforms(image, image_heatmaps)
        
        self.display_landmarks(image, image_heatmaps)
        
        output = { 'image' : image, 'image_heatmaps' : image_heatmaps}
        #exit()
        
        return output
        
    
def main():
    
    dataset_path = 'C:\\Users\\nikhar.m.1\\Documents\\FaceLandmarkDetection\\Face Landmark Detection\\data\\'
    tforms = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(200)])
    #dataset = faceDB300NW(dataset_path, 'full_test_set', tforms)
    
    train_dl = DataLoader((faceDB300NW(dataset_path, 'nikhar', tforms)),
                                        batch_size = 1, shuffle = False,
                                        pin_memory = True,
                                        drop_last = True)
                                        
    for i, data in enumerate(train_dl):
        image, hms = data['image'], data['image_heatmaps']
        

#%%

if __name__ == '__main__':
    main()