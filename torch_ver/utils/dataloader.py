import os
import pandas as pd
import torch.utils.data
import torchvision.transforms.functional as FT
from torchvision.io import read_image
from typing import Dict, List, Tuple, Any

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



def parse_from_target(target):
    voc_dict: Dict = {}

    classes: List = []
    bboxes: List = []

    for i in range(len(target['annotation']['object'])):
        bbox: List = []
        _class = target['annotation']['object'][i]['name']
        bndbox = target['annotation']['object'][i]['bndbox']
        width = target['annotation']['size']['width']
        height = target['annotation']['size']['height']
        for key, val in bndbox.items():
            bbox.append(int(val))
            classes.append(_class)
        bboxes.append(bbox)

    voc_dict['labels'] = classes
    voc_dict['bboxes'] = bboxes
    voc_dict['size'] = (int(width), int(height))
    return voc_dict



def resize(image, bboxes, image_size, dims: Tuple=(300, 300)):
    width, height = image_size[0], image_size[1]

    # Resize image
    re_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([width, height, width, height]).unsqueeze(0)
    re_bboxes: List = []
    for bbox in bboxes:
        bbox = torch.tensor(bbox)
        re_bbox = bbox / old_dims  # percent coordinates
        re_bbox = re_bbox.tolist()[0]

        for i, rate in enumerate(re_bbox):
            if i % 2 == 0:
                re_bbox[i] = rate * dims[0]
            elif i % 2 == 1:
                re_bbox[i] = rate * dims[1]
        re_bboxes.append(re_bbox)
    re_bboxes = torch.tensor(re_bboxes)

    return re_image, re_bboxes