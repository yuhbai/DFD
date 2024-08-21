import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from data.perlin import rand_perlin_2d_np
import random
import json
import torchvision.transforms as transforms
from PIL import Image
import cv2

def FD(img):
    img_resize = cv2.pyrDown(img)
    temp_pyrUp = cv2.pyrUp(img_resize)
    #pdb.set_trace()
    temp_lap = cv2.subtract(img, temp_pyrUp)

    return temp_lap, temp_pyrUp

class MVTecTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape
        self.final_processing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape, self.resize_shape))
            mask = cv2.resize(mask, dsize=(self.resize_shape, self.resize_shape))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        lab, res = FD(image)
        lab, res = self.final_processing(lab), self.final_processing(res)

        mask = np.transpose(mask, (2, 0, 1))
        image = self.final_processing(image)
        return image, mask, lab, res

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask, lab, res = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask, lab, res = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx, 'lab':lab, 'res':res}

        return sample



class MVTecTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, k_shot, num = 50, resize_shape=224, use_mask=None, bg_reverse=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.k_shot = k_shot
        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))
        self.image_paths =  random.sample(self.image_paths,k_shot)
        self.use_mask = use_mask
        self.bg_reverse = bg_reverse
        self.num = num
        # with open(str(k_shot) + '.json', 'w') as file:
        #     for item in self.image_paths:
        #         file.write(json.dumps(item) + '\n')
            
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.aug = iaa.Sequential([iaa.GammaContrast((0.5,2.0),per_channel=True),
                                   iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                                   iaa.pillike.EnhanceSharpness(),
                                   iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                                   iaa.Affine(rotate=(-90, 90)),]
                                  )
        self.final_processing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        self.aug_images = []
        for idx in range(len(self.image_paths)):
            for i in range(self.num):
                anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
                anomaly_mask, has_anomaly, lab, res, lab_aug, res_aug, image, image_aug = self.transform_image(self.image_paths[idx],
                                                                                self.anomaly_source_paths[anomaly_source_idx])
                
                sample = {"anomaly_mask": anomaly_mask,'has_anomaly': has_anomaly, 'lab':lab, 'res':res, 'lab_aug':lab_aug, 'res_aug':res_aug, 'image':image, 'image_aug':image_aug}
                self.aug_images.append(sample)
       
    def __len__(self):
        return self.num*self.k_shot


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug
    
    def foreground(self, image):

        _, img_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_mask = img_mask.astype(np.bool).astype(np.int)
        
        if self.bg_reverse:
            foreground_mask = img_mask
        else:
            foreground_mask = -(img_mask - 1)

        return foreground_mask


    def augment_image(self, image, img_mask, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape, self.resize_shape))

        anomaly_img_augmented = aug(image=anomaly_source_img) 
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        
        perlin_noise = rand_perlin_2d_np((self.resize_shape, self.resize_shape), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))*img_mask
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.8:
            image = image.astype(np.float32)
            augmented_image = image
            return augmented_image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
        return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape, self.resize_shape))  
        if self.use_mask:
            bg_color = image[0][0, 0]
            rot_img = iaa.Sequential([iaa.Affine(rotate=(-90,90), mode='constant', cval=bg_color)])
            image = rot_img(image=image)
            gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            img_mask = self.foreground(image=gray_image)
        else:
            image = self.rot(image=image)
            img_mask = np.ones((self.resize_shape, self.resize_shape))
        
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, img_mask, anomaly_source_path)


        lab_aug, res_aug = FD(augmented_image)
        lab, res = FD(image)
        lab_aug, res_aug = self.final_processing(lab_aug),  self.final_processing(res_aug)
        lab, res = self.final_processing(lab),  self.final_processing(res)
        image = self.final_processing(image)
        image_aug = self.final_processing(augmented_image)
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
 
        return anomaly_mask, has_anomaly, lab, res, lab_aug, res_aug, image, image_aug

    def __getitem__(self, idx):

        idx = torch.randint(0, len(self.aug_images), (1,)).item()
        sample = self.aug_images[idx]
        
        return sample



class VisATestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.JPG"))
        self.resize_shape=resize_shape
        self.final_processing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape, self.resize_shape))
            mask = cv2.resize(mask, dsize=(self.resize_shape, self.resize_shape))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        lab, res = FD(image)
        lab, res = self.final_processing(lab), self.final_processing(res)

        mask = np.transpose(mask, (2, 0, 1))
        image = self.final_processing(image)
        return image, mask, lab, res

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask, lab, res = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+".png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask, lab, res = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx, 'lab':lab, 'res':res}

        return sample



class VisATrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, k_shot, num = 50, resize_shape=224):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.k_shot = k_shot
        self.num = num
        random.seed(0)
        self.image_paths = sorted(glob.glob(root_dir+"/*.JPG"))
        self.image_paths =  random.sample(self.image_paths,k_shot)
            
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.aug = iaa.Sequential([iaa.GammaContrast((0.5,2.0),per_channel=True),
                                   iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                                   iaa.pillike.EnhanceSharpness(),
                                   iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                                   iaa.Affine(rotate=(-90, 90)),]
                                  )
        self.final_processing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        self.aug_images = []
        for idx in range(len(self.image_paths)):
            for i in range(self.num):
                anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
                anomaly_mask, has_anomaly, lab, res, lab_aug, res_aug, image, image_aug = self.transform_image(self.image_paths[idx],
                                                                                self.anomaly_source_paths[anomaly_source_idx])
                
                sample = {"anomaly_mask": anomaly_mask,'has_anomaly': has_anomaly, 'lab':lab, 'res':res, 'lab_aug':lab_aug, 'res_aug':res_aug, 'image':image, 'image_aug':image_aug}
                self.aug_images.append(sample)
       
    def __len__(self):
        return self.num*self.k_shot


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug
    
    def foreground(self, image):

        _, img_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_mask = img_mask.astype(np.bool).astype(np.int)
        
        if self.bg_reverse:
            foreground_mask = img_mask
        else:
            foreground_mask = -(img_mask - 1)

        return foreground_mask


    def augment_image(self, image, img_mask, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape, self.resize_shape))

        anomaly_img_augmented = aug(image=anomaly_source_img) 
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        
        perlin_noise = rand_perlin_2d_np((self.resize_shape, self.resize_shape), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))*img_mask
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.7:
            image = image.astype(np.float32)
            augmented_image = image
            return augmented_image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
        return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape, self.resize_shape))  
        image = self.rot(image=image)
        img_mask = np.ones((self.resize_shape, self.resize_shape))
        
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, img_mask, anomaly_source_path)


        lab_aug, res_aug = FD(augmented_image)
        lab, res = FD(image)
        lab_aug, res_aug = self.final_processing(lab_aug),  self.final_processing(res_aug)
        lab, res = self.final_processing(lab),  self.final_processing(res)
        image = self.final_processing(image)
        image_aug = self.final_processing(augmented_image)
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
 
        return anomaly_mask, has_anomaly, lab, res, lab_aug, res_aug, image, image_aug

    def __getitem__(self, idx):

        idx = torch.randint(0, len(self.aug_images), (1,)).item()
        sample = self.aug_images[idx]
        
        return sample
