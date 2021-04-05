import os.path
import torch.utils.data as data
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import numpy as np
import torch


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

class RemovalDataset(BaseDataset):
    def initialize(self, opt):

        self.root = opt.dataroot
        self.path_mixture = os.path.join(self.root, 'syn')
        self.path_ground = os.path.join(self.root, 't')
        self.path_reflection = os.path.join(self.root, 'rblur')
        self.path_map = os.path.join(self.root, 'W')
        self.norm_path = os.path.join(self.root, 'para_zq.txt')
        
        # files = os.listdir(self.path_mixture)
        # self.num_images = len(files)

        lines = [line.rstrip() for line in open(self.norm_path, 'r')]
        random.seed(1234)
        random.shuffle(lines)
        self.num_images = len(lines)
        print(self.num_images)
        print(self.num_images)
        print(self.num_images)
        print(self.num_images)
        self.datapair = []
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            norm = np.array([float(split[-3]),float(split[-2]),float(split[-1])])
            self.datapair.append([filename,norm])


    def get_transforms_0(self, img, i, j):
        img = transforms.functional.crop(img, i, j, 256, 256)

        return img

    def get_transforms_1(self, img):
        transform = transforms.CenterCrop(512)
        img = transform(img)

        return img

    def get_transforms_2(self, img):
        transform_list = []
        # transform_list.append(transforms.Resize(80))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),
                                                   (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)
        img = transform(img)

        return img

    def __getitem__(self, index):
        filename,norm = self.datapair[index]
        C_img = Image.open(os.path.join(self.path_mixture, filename+'.jpg')).convert('RGB')
        A_img = Image.open(os.path.join(self.path_ground, filename+'.jpg')).convert('RGB')
        B_img = Image.open(os.path.join(self.path_reflection, filename+'.jpg')).convert('RGB')
        W_np = np.load(os.path.join(self.path_map, filename+'.npy'))
        # W_np = W_np.reshape([320,480])
        # W_np_s = W_np[::4,::4]

        C = self.get_transforms_2(C_img)
        A = self.get_transforms_2(A_img)
        B = self.get_transforms_2(B_img)
        W = torch.from_numpy(W_np).view(1, 320, 480).float()
        norm=torch.from_numpy(norm).float()

        return {'A': A, 'B': B, 'C': C, 'W': W, 'filename': filename, 'norm': norm}

    def __len__(self):
        return self.num_images

    def name(self):
        return 'RemovalDataset'
    
    
 class RemovalDatasetTest(BaseDataset):
    def initialize(self, opt):

        self.root = opt.dataroot
        self.path_mixture = os.path.join(self.root, 'data')
        self.norm_path = os.path.join(self.root, 'gt.txt')
        
        files = os.listdir(self.path_mixture)
        # self.num_images = len(files)

        lines = [line.rstrip() for line in open(self.norm_path, 'r')]
        random.seed(1234)
        random.shuffle(lines)
        self.num_images = len(lines)
        print(self.num_images)
        self.datapair = []
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            norm = np.array([float(split[-3]),float(split[-2]),float(split[-1])])
            self.datapair.append([filename,norm])


    def get_transforms_0(self, img, i, j):
        img = transforms.functional.crop(img, i, j, 256, 256)

        return img

    def get_transforms_1(self, img):
        transform = transforms.CenterCrop(512)
        img = transform(img)

        return img

    def get_transforms_2(self, img):
        transform_list = []
        # transform_list.append(transforms.Resize(80))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),
                                                   (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)
        img = transform(img)

        return img

    def __getitem__(self, index):
        filename,norm = self.datapair[index]
        C_img = Image.open(os.path.join(self.path_mixture, filename+'.jpg')).convert('RGB')
        # W_np = W_np.reshape([320,480])
        # W_np_s = W_np[::4,::4]

        C = self.get_transforms_2(C_img)
       
        norm=torch.from_numpy(norm).float()

        return {'C': C, 'filename': filename, 'norm': norm}

    def __len__(self):
        return self.num_images

    def name(self):
        return 'RemovalDatasetTest'
