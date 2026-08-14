import os
import math
import torch
import random
import json
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import albumentations as A
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import skimage as ski
import pandas as pd


class ZiramF0Dataset(Dataset):
    def __init__(self, 
                 img_type, 
                 train=False,
                 val=False,
                 test=False,
                 apply_transform=True,
                 return_mask=False, 
                 masks_root_path=None
                ):
        super(ZiramF0Dataset, self).__init__()

        self.train = train
        self.val = val
        self.test = test

        self.apply_transform = apply_transform
        self.img_type = img_type
        self.train_suffix = "train" if self.train else "test"
        self.return_mask = return_mask
        self.masks_root_path = masks_root_path
        
        if self.img_type in {"brightfield", "fluorescent"}: # For data in path /nfs/research/birney/users/esther/medaka-ziram/data/f0/
            # Root path where the raw images are stored
            self.images_root_path = f"/nfs/research/birney/users/esther/medaka-ziram/data/f0/{self.img_type}_{self.train_suffix}/"

            # Read in the associated metadata
            self.metadata = pd.read_csv("/nfs/research/birney/users/esther/medaka-ziram/data/f0/Gronske_F0_Data.csv", index_col=0)
            
            # Get the full path of the images, selecting only for those that have a valid entry in provided metadata
            self.images = sorted([self.images_root_path + i for i in os.listdir(self.images_root_path) if 
                                    (i.endswith('.tif') and (os.path.splitext(i)[0] in self.metadata["CO4"].values if self.img_type == "fluorescent" else os.path.splitext(i)[0] in self.metadata["CO6"].values))])
        
        elif self.img_type == "full_dataset": # For data in path /nfs/research/birney/users/esther/medaka-ziram/data/full_dataset/
            # self.metadata = pd.read_csv("/nfs/research/birney/users/esther/medaka-ziram/data/Ziram_Full_Dataset_balanced.csv") # Only F0 in training set
            self.metadata = pd.read_csv("/nfs/research/birney/users/esther/medaka-ziram/data/Ziram_Full_Dataset_trainrebalancedF2_QC.csv") # F0 and F2 in training set
            self.metadata['image_name'] = self.metadata['image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

            # Only F0 in training set
            # if self.train:
            #     self.images_root_path = "/nfs/research/birney/users/esther/medaka-ziram/data/full_dataset/train/"
            # elif self.val:
            #     self.images_root_path = "/nfs/research/birney/users/esther/medaka-ziram/data/full_dataset/val/"
            # elif self.test:
            #     self.images_root_path = "/nfs/research/birney/users/esther/medaka-ziram/data/full_dataset/test/"

            # F0 and F2 in training set
            if self.train:
                self.images_root_path = "/nfs/research/birney/users/esther/medaka-ziram/data/full_dataset_trainrebalanced_qc/train/"
            elif self.val:
                self.images_root_path = "/nfs/research/birney/users/esther/medaka-ziram/data/full_dataset_trainrebalanced_qc/val/"
            elif self.test:
                self.images_root_path = "/nfs/research/birney/users/esther/medaka-ziram/data/full_dataset_trainrebalanced_qc/test/"

            self.images = sorted([self.images_root_path + i for i in os.listdir(self.images_root_path) if i.endswith('.tif')])
        
        else:
            raise Exception("img_type must be either 'brightfield' or 'fluorescent'")
        

        if self.return_mask:
            if self.masks_root_path is None:
                raise Exception("if return_mask is True then masks_root_path must be provided")
            self.masks = sorted([self.masks_root_path + i for i in os.listdir(self.masks_root_path) if i.lower().endswith(('.tif'))])

            # Not all images had valid masks outputted; remove images without a corresponding mask from the list
            images_filename_only = [os.path.basename(f) for f in self.images]
            masks_filename_only  = [os.path.basename(f) for f in self.masks]
            
            self.images_to_remove_idx = [i for i, x in enumerate(images_filename_only) if x not in masks_filename_only]
            self.images = [x for i, x in enumerate(self.images) if i not in self.images_to_remove_idx]

            # Also load bounding box coordinates
            with open(f"{self.masks_root_path}/annotations_rough_bbox.json", 'r', encoding='utf-8') as file:
                bbox_file = json.load(file)
            
            bbox_ordered = {key: bbox_file[key] for key in self.masks}
            bbox_regionprops_to_sam2 = {key: [x1, y1, x2, y2] for key, [y1, x1, y2, x2] in bbox_ordered.items()}
            self.bbox = [[v] for k, v in bbox_regionprops_to_sam2.items()]


        ### Order the metadata rows according to entries of self.images
        ### (This is mainly for ease of plotting after training)
        # Get name of the images only (no path or extension)
        self.images_basename = [os.path.splitext(os.path.basename(i))[0] for i in self.images]

        if self.img_type in {"brightfield", "fluorescent"}:
            # Remove rows in metadata that are not in images
            self.metadata = self.metadata[self.metadata['CO4'].isin(self.images_basename)] if self.img_type == "fluorescent" else self.metadata[self.metadata['CO6'].isin(self.images_basename)]

            self.metadata_ordered = self.metadata.set_index("CO4" if self.img_type == "fluorescent" else "CO6")
            self.metadata_ordered = self.metadata_ordered.loc[self.images_basename]
        
        elif self.img_type == "full_dataset":
            # Remove rows in metadata that are not in images
            self.metadata = self.metadata[self.metadata['image_name'].isin(self.images_basename)]
            self.metadata_ordered = self.metadata.set_index("image_name")
            self.metadata_ordered = self.metadata_ordered.loc[self.images_basename]


    def __len__(self):
        return(len(self.images))


    def __getitem__(self, index):
        # brightfield images are uint16 in their raw form
        # SAM2 expects pixel values in the range of [0, 255] (uint8)
        img_basename = os.path.splitext(os.path.basename(self.images[index]))[0]

        if self.img_type == "brightfield":
            img = ski.util.img_as_ubyte(ski.io.imread(self.images[index])) # (2048, 2048)
            img_metadata = self.metadata.loc[self.metadata['CO6'] == img_basename]
        elif self.img_type == "fluorescent":
            # TODO: Implement this?
            img_metadata = self.metadata.loc[self.metadata['CO4'] == img_basename]
            pass
        elif self.img_type == "full_dataset":
            img = ski.util.img_as_ubyte(ski.io.imread(self.images[index]))
            img_metadata = self.metadata.loc[self.metadata['image_name'] == img_basename]
        else:
            raise Exception("img_type must be either 'brightfield' or 'fluorescent'")
        
        if self.img_type in {"brightfield", "fluorescent"}:
            # Create metadata dict
            metadata_dict = {
                "line": img_metadata["line_x"].values.item(),
                "exposure_type": img_metadata["exposure_type"].values.item(),
                "kinked": img_metadata["kinked"].values.item(),
                "num_kinks": img_metadata["#_kinks"].values.item(),
                "severity": img_metadata["severity_score"].values.item(),
                "date": img_metadata["date"].values.item(),
                "plate": img_metadata["plate_#"].values.item(),
                "well": img_metadata["well_#"].values.item(),
            }
        elif self.img_type == "full_dataset":
            metadata_dict = {
                "generation": img_metadata["generation"].values.item(),
                "severity_score": img_metadata["severity_score"].values.item(),
                "severity_score_adjusted": img_metadata["severity_score_adjusted"].values.item(),
                "kinks_number": img_metadata["kinks_number"].values.item(),
                "plate": img_metadata["plate"].values.item(),
                "well": img_metadata["well"].values.item(),
                "line": img_metadata["line"].fillna('').values.item(),
                "exposure_type": img_metadata["exposure_type"].fillna('').values.item(),
                "cross": img_metadata["cross"].fillna('').values.item(),
                "tank": img_metadata["tank"].fillna(-1).values.item(),
            }
        
        if self.apply_transform: # For training
            self.transform = A.Compose([
                A.RandomCrop(height=1536, width=1536),
                A.Resize(height=352, width=352),
                A.RandomBrightnessContrast(p=1.0),
                A.Normalize(
                    normalization="image",
                    max_pixel_value=1.0,
                    p=1.0
                ),
                A.SquareSymmetry(p=1.0)
            ])
        else: # For inference
            self.transform = A.Compose([
                A.CenterCrop(height=1536, width=1536),
                A.Resize(height=352, width=352),
                A.Normalize(
                    normalization="image",
                    max_pixel_value=1.0,
                    p=1.0
                )
            ])

        
        if self.return_mask:
            mask = ski.util.img_as_ubyte(ski.io.imread(self.masks[index]))
            
            augmented = self.transform(image = img, mask = mask)
            img, mask = augmented['image'], augmented['mask']

            return transforms.ToTensor()(np.array(img)), mask, np.array(self.bbox[index]), self.images[index], self.masks[index], metadata_dict
        else:
            augmented = self.transform(image=img)
            img = augmented['image']

            return transforms.ToTensor()(np.array(img)), torch.tensor(0.0), np.array(0.0), self.images[index], '', metadata_dict