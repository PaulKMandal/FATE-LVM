import os
import numpy as np
import pandas as pd
from federatedml.nn.dataset.base import Dataset
from federatedml.util import LOGGER
from federatedml.nn.dataset.image import ImageDataset
import torchvision.transforms as T
import torch
from PIL import Image
from pycocotools.coco import COCO
from transformers import DetrImageProcessor, DetrFeatureExtractor
import torchvision

class TokenizerDetrDataset(Dataset):

    """

    """

    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        LOGGER.info("pre create processor")
        self.processor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50", size=500, max_size=600)
        LOGGER.info("post create processor")
        #resize dim
        #self.transform = T.Resize((224,224))

    #Need to edit. Also need to figure out how to incorporate collate_fn
    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]

        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)

        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)

        #LOGGER.info("coco_annotation")
        #LOGGER.info(coco_annotation)
        
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]

        # open the input image
        img = Image.open(os.path.join(self.root, path))
        # number of objects in the image
        num_objs = len(coco_annotation)
        

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        
        #MAKING EDITS HERE
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        if boxes == []:
            boxes = torch.zeros(0,4)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Labels (In my case, I only one class: target class or background)
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)

        areas=[]
        labels=[]
        
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
            if coco_annotation[i]['category_id'] > 80:
                labels.append(0)   
            else:
                labels.append(coco_annotation[i]['category_id']-1)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)

                # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["image"] = img
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        #LOGGER.info("my_annotation")
        #LOGGER.info(my_annotation)
        
        return my_annotation

    def __len__(self):
        return len(self.ids)
        
    def get_dataset(self):
        return self.dataset

    def get_classes(self):
        return self.dataset.get_classes()
        
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)
