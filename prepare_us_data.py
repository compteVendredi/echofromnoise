import os
import shutil
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from natsort import natsorted
from torchio import RandomElasticDeformation
import torch

def create_folders(base_path):
    annotations_path = os.path.join(base_path, "annotations")
    sector_annotations_path = os.path.join(base_path, "sector_annotations")
    images_path = os.path.join(base_path, "images")
    for folder in [annotations_path, images_path, sector_annotations_path]:
        for subfolder in ["training", "testing", "validation"]:
            os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

def load_mat(mat_file):
    data = scipy.io.loadmat(mat_file)
    mask = data["mask"]
    mask = Image.fromarray(mask)
    mask = mask.resize((256, 256), Image.NEAREST)  
    mask = mask.crop((0, 0, 256, 256))  
    return np.array(mask).astype(np.uint8)

def process_directory(src_root, dst_root):
    for category, dirs in {"training": ["1"], "testing":["2"], "validation": ["3"]}.items():
        for d in dirs:
            src_dir = os.path.join(src_root, d)
            img_dst_dir = os.path.join(dst_root, "images", category)
            ann_dst_dir = os.path.join(dst_root, "annotations", category)
            sector_ann_dst_dir = os.path.join(dst_root, "sector_annotations", category)
            os.makedirs(img_dst_dir, exist_ok=True)
            os.makedirs(ann_dst_dir, exist_ok=True)
            os.makedirs(sector_ann_dst_dir, exist_ok=True)
            
            img_path = os.path.join(src_dir, "bmode_GT.png")
            img = Image.open(img_path)
            img = img.resize((256, 256))  
            img = img.crop((0, 0, 256, 256))  
            mask_path = os.path.join(src_dir, "mask", "mask.mat")

            if category != "testing":
                for i in range(5):
                    img.save(os.path.join(img_dst_dir, str(i)+"_"+d + ".png"))          
                    sector_labelmap, labelmap = add_labels2camus(load_mat(mask_path), np.array(img), True)
                    labelmap = Image.fromarray(labelmap)
                    sector_labelmap = Image.fromarray(sector_labelmap)
                    labelmap.save(os.path.join(ann_dst_dir, str(i)+"_"+d + ".png"))
                    sector_labelmap.save(os.path.join(sector_ann_dst_dir, str(i)+"_"+d + ".png"))
            else:
                img.save(os.path.join(img_dst_dir, str(0)+"_"+d + ".png"))     
                sector_labelmap, labelmap = add_labels2camus(load_mat(mask_path), np.array(img), False)
                labelmap = Image.fromarray(labelmap)
                sector_labelmap = Image.fromarray(sector_labelmap)
                labelmap.save(os.path.join(ann_dst_dir, str(0)+"_"+d + ".png"))
                sector_labelmap.save(os.path.join(sector_ann_dst_dir, str(0)+"_"+d + ".png"))




#####Modification of augment_camus_labels.py

def postprocess_tensor(in_tensor):
    transform = transforms.ToPILImage()
    img_out = transform(in_tensor[0])
    return img_out


def augment_img_tensor(label_tensor):
    affine_degrees = (-5, 5)
    affine_translate = (0, 0.05)
    affine_scale = (0.8, 1.05)
    affine_shear = 5

    elastic_num_control_points = (10, 10, 4)#(10, 10, 4)
    elastic_locked_borders = 1
    elastic_max_displacement = (0, 18, 18)#(0, 30, 30)
    elastic_image_interpolation = 'nearest'

    img_transforms = transforms.Compose(
        [           
            transforms.RandomAffine(
                degrees=affine_degrees,
                translate=affine_translate,
                scale=affine_scale,
                shear=affine_shear,
            ),
            RandomElasticDeformation(
                num_control_points=elastic_num_control_points,
                locked_borders=elastic_locked_borders,
                max_displacement=elastic_max_displacement,
                image_interpolation=elastic_image_interpolation
            )
        ])

    img_out = img_transforms(label_tensor)

    img_out = postprocess_tensor(img_out)
    return img_out


def add_labels2camus(in_label_img, in_real_img, augment_img):
    in_real_img = np.asarray(in_real_img)
    out_label_img = np.copy(in_label_img)
    cone_label = np.zeros_like(in_label_img)+1
    #cone_label[in_real_img != 0] = 1  # create cone mask from real image

    if augment_img:
        out_label_img = augment_img_tensor(transforms.functional.to_tensor(out_label_img).unsqueeze(0))

    unclipped_labels = np.copy(out_label_img)
    out_label_img *= cone_label  # clips labels that are outside the cone
    out_label_img[out_label_img > 1] += 1  # make space for new label to match already generated datasets
    out_label_img[(out_label_img == 0) & (cone_label == 1)] = 2

    return out_label_img, unclipped_labels.astype(np.uint8)


#####

base_src = "US_data/simu"
base_dst = "prepared_US_data"
create_folders(base_dst)
process_directory(base_src, base_dst)
