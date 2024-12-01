import os
import rasterio
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from albumentations import ShiftScaleRotate



class CustomDataset(Dataset):
    def __init__(self, root_dir, test=False, use_dem=False,dem_derives=False,source=None,target_value=0,NDVI=False,train_all=False):
        """
        Initlize the dataset
        root_dir (str): path of the dataset, containing four subsets named 'images', 'labels', 'train', 'test', 'DEM_derives'(optional)
        train_all: load all data without splitting into training or testing sets
        test (bool): the currently loaded dataset is either the training or testing set. It will load samples from the 'train' or 'test' folder based on the corrsponding input and apply the appropriate data augmentation
        use_dem (bool): whether to use DEM as an additional input feature (if subfolder 'DEM_derives' exists)
        NDVI (bool): whether to use NDVI as an additional input feature (if the image file contains four channels: RGB and NIR)
        source (str): if source=="gaofen", the image's origional (B,G,R,NIR) channel order will be adjusted  to (R,G,B,NIR)
        target_value (int): the value representing the foreground in the labeled samples within the labels foder. The mask value will be transformed based on this value.
        """
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir
        self.train_all=train_all
        self.use_dem = use_dem
        self.NDVI=NDVI
        self.dem_derives=dem_derives
        self.test = test
        self.source=source
        self.target_value=target_value
        self.terrain_param_folder_names=[]

        if self.train_all:
            self.image_dir=os.path.join(self.root_dir,'images')
            self.label_dir=os.path.join(self.root_dir,'labels')
            if self.use_dem:
                self.dem_dir=os.path.join(self.root_dir,'DEM_derives')
                if self.dem_derives:
                    self.terrain_params_subfolders=sorted(os.listdir(self.dem_dir))
                else:
                    self.dem_files=sorted(os.listdir(os.path.join(self.dem_dir,'dem')))
        else:
            sub_dir = 'test' if self.test else 'train'
            self.image_dir = os.path.join(self.root_dir, sub_dir, 'images')
            self.label_dir = os.path.join(self.root_dir, sub_dir, 'labels')
            if self.use_dem:
                self.dem_dir = os.path.join(self.root_dir, sub_dir, 'DEM_derives')
                if self.dem_derives:
                    self.terrain_params_subfolders=sorted(os.listdir(self.dem_dir))
                else:
                    self.dem_files=sorted(os.listdir(os.path.join(self.dem_dir,'dem')))
            else:
                self.dem_dir=None
                self.dem_files=None

        self.image_files = sorted(os.listdir(self.image_dir))
        self.label_files = sorted(os.listdir(self.label_dir))
        if len(self.image_files)!=len(self.label_files):
            raise ValueError(f'error : the number of images is not equal to the labels')

        self.train_transform = A.Compose([
            A.Resize(height=256, width=256),
            A.HorizontalFlip(p=0.5),   
            A.VerticalFlip(p=0.5),     
            A.RandomResizedCrop(width=256, height=256, scale=(0.5, 0.8), ratio=(0.75, 1.333), p=0.3),
            A.Rotate(limit=45, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=0),  
            A.GaussNoise(var_limit=(0.0005, 0.005), p=0.2),    
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),   
            ToTensorV2(transpose_mask=True)
        ])

        self.test_transform = A.Compose([
            A.Resize(height=256, width=256),
            ToTensorV2(transpose_mask=True)
        ])

    def __len__(self):
        return len(self.image_files)

    def _load_image(self, file_path):
        if file_path.lower().endswith('.png'):
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  
            if img.ndim == 2: 
                img = np.expand_dims(img, axis=-1)  
        elif file_path.lower().endswith('.tif') or file_path.lower().endswith('.tiff'):
            with rasterio.open(file_path) as src:
                img = src.read()
                img = np.moveaxis(img, 0, -1) 
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        return img
    
    def _load_dem_and_derives(self,idx,img_size):
        terrain_params=[]
        if self.dem_derives:
            for folder in self.terrain_params_subfolders:
                param_file=os.path.join(self.dem_dir,folder,self.image_files[idx])
                param_file=os.path.splitext(param_file)[0]+'.tif'

                param=self._load_image(param_file)
                param=self._process_param(param,img_size)
                terrain_params.append(param)
                if folder not in self.terrain_param_folder_names:
                    self.terrain_param_folder_names.append(folder)

        else:
            dem_file=os.path.join(self.dem_dir,'dem',self.dem_files[idx])
            param=self._load_image(dem_file)
            param=self._process_param(param,img_size)
            terrain_params.append(param)
            self.terrain_param_folder_names.append('dem')
        return np.concatenate(terrain_params,axis=-1)
    
    def _process_param(self,param,img_size):
        if param.shape[:2]!=img_size:
            param=cv2.resize(param,(img_size[1],img_size[0]),interpolation=cv2.INTER_LINEAR)
        if param.ndim==2:
            param=np.expand_dims(param,axis=-1)
        param=param.astype(np.float32)
        param_min=param.min()
        param_max=param.max()
        if param_max>param_min:
            param=(param-param_min)/(param_max-param_min)
        else:
            param=np.zeros_like(param,dtype=np.float32)

        return param
    
    def _calculate_ndvi(self,img):
        if img.shape[2]>=3:
            red=img[:,:,0]
            nir=img[:,:,3]
            ndvi=(nir-red)/(nir+red+1e-10)
            ndvi=np.clip(ndvi,-1,1)
            return ndvi
        else:
            raise ValueError('Image does not fit the number of channels!!')
        



    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        label_name = os.path.join(self.label_dir, self.label_files[idx])
        img = self._load_image(img_name)
        mask = self._load_image(label_name)

        if img.max()>1:
            img=img.astype(np.float32)/255.0
        if mask.max()>1:
            mask=mask.astype(np.float32)/255.0
        if self.target_value== 0:
            mask = 1 - mask  
        elif self.target_value==1: 
            mask=mask
        if self.source == 'gaofen' and img.shape[2] >= 3:
            img[:, :, :3] = img[:, :, [2, 1, 0]]  
        elif self.source is not None and self.source!='gaofen':
            raise ValueError(f'Unsupported source value: {self.source}')
        
        if self.NDVI:
            ndvi=self._calculate_ndvi(img)
            ndvi=np.expand_dims(ndvi,axis=-1)
            img=np.concatenate((img,ndvi),axis=-1)

        # 读取DEM（如果使用）
        if self.use_dem:
            img_size=img.shape[:2]
            terrain_params=self._load_dem_and_derives(idx,img_size)
            img=np.concatenate((img,terrain_params),axis=-1)
 
        h,w=mask.shape[:2]
        binary_mask=np.zeros((h,w,2),dtype=np.float32)
        binary_mask[...,0]=(mask[...,0]==0).astype(np.float32)
        binary_mask[...,1]=(mask[...,0]==1).astype(np.float32)
        mask=binary_mask

        img_channels=img.shape[2]
        mask_channels=mask.shape[2]
        if img_channels > mask_channels:
            new_channel = np.ones((img.shape[0], img.shape[1], 1)) 
            expanded_mask = np.concatenate([mask] + [new_channel] * (img_channels - mask_channels), axis=-1)
            mask = expanded_mask
        if not self.test:
            transformed = self.train_transform(image=img, mask=mask)
        else:
            transformed = self.test_transform(image=img, mask=mask)

        img = transformed["image"].float()
        mask = (transformed["mask"]).float()
        mask=mask[:mask_channels]

        return img, mask, img_name, label_name
    
class RGB_Dataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        img, mask,img_path,mask_path = self.dataset[idx] 
        if img.shape[0] >= 4:
            img = img[:3, :, :] 

        return img, mask,img_path,mask_path

    def __len__(self):
        return len(self.dataset)
    

class Fliter_Dataset:
    def __init__(self, dataset, channel_to_remove):
        self.dataset = dataset
        self.channel_to_remove = channel_to_remove  

    def __getitem__(self, idx):
        img, mask, img_path, mask_path = self.dataset[idx]  
        if img.shape[0] > self.channel_to_remove:  
            img = img[[i for i in range(img.shape[0]) if i != self.channel_to_remove], :, :]  
        return img, mask, img_path, mask_path

    def __len__(self):
        return len(self.dataset)




