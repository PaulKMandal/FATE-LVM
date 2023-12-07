import torch
from federatedml.nn.dataset.base import Dataset
from torchvision.datasets import ImageFolder
from fate_llm.dataset.detr_dataset import TokenizerDetrDataset, get_transform
from torchvision import transforms
import numpy as np
from federatedml.util import LOGGER
from transformers import DetrImageProcessor,DetrFeatureExtractor
import json
class DetrTokenizer(Dataset):

    def __init__(self, center_crop=False,
                 generate_id_from_file_name=True, file_suffix='.jpg',
                 return_label=True, float64=False, label_dtype='long'):

        super(DetrTokenizer, self).__init__()
        self.annotations: AnnotationDataset = None
        self.return_label = return_label
        self.generate_id_from_file_name = generate_id_from_file_name
        self.file_suffix = file_suffix
        self.float64 = float64
        self.dtype = torch.float32 if not self.float64 else torch.float64
        avail_label_type = ['float', 'long', 'double']
        self.sample_ids = None
        self.processor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50", size=500, max_size=600)

        assert label_dtype in avail_label_type, 'available label dtype : {}'.format(
            avail_label_type)
        if label_dtype == 'double':
            self.label_dtype = torch.float64
        elif label_dtype == 'long':
            self.label_dtype = torch.int64
        else:
            self.label_dtype = torch.float32

    def load(self, folder_path, annotation_path):

        # read image from folders
        self.ann_folder = annotation_path
        with open(self.ann_folder, 'r') as f:
            self.coco = json.load(f)
        self.annotations = TokenizerDetrDataset(root=folder_path,annotation=annotation_path,transforms=get_transform())

    def load(self, path):

        # read image from folders

        #self.annotations = AnnotationDataset(root=path+'/img/',annotation=path+'/coco.json', transforms=get_transform())
        self.ann_folder = path+'coco.json'
        with open(self.ann_folder, 'r') as f:
            self.coco = json.load(f)
        self.annotations = TokenizerDetrDataset(root=path+'img/',annotation=path+'coco.json', transforms=get_transform())

    def __getitem__(self, item):
        """
        ann_info = self.coco['annotations'][item] if "annotations" in self.coco else self.coco['images'][item]

        LOGGER.info("ann_info")
        LOGGER.info(ann_info)
        
        res = self.annotations.__getitem__(item)
        img = res['image']
        ann_info = {'boxes': res['boxes'],'labels': res['labels'],'image_id': res['image_id'],'area': res['area'],'iscrowd':res['iscrowd']}
        LOGGER.info(res)
        
        encoding = self.processor(images=img, annotations=ann_info, masks_path=self.ann_folder, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0]

        LOGGER.info("post encoding in tokenizer")
        LOGGER.info(encoding)
        LOGGER.info(pixel_values)
        LOGGER.info(target)
        """
        return self.annotations.__getitem__(item)

    def __len__(self):
        return self.annotations.__len__()

    def __repr__(self):
        return self.annotations.__repr__()

    def get_classes(self):
        return self.annotations.__len__()

    def get_sample_ids(self):
        return self.sample_ids
        
    def collate_fn(self, batch):

        ret_label = {}
        #ret_label['boxes'] = [item["boxes"] for item in batch]
        #ret_label['class_labels'] = [item["labels"] for item in batch]
        #ret_label['image_id'] = [item["image_id"] for item in batch]
        #ret_label['area'] = [item["area"] for item in batch]
        #ret_label['iscrowd'] = [item["iscrowd"] for item in batch]

        ret_label['boxes'] = torch.tensor(batch[0]["boxes"])
        ret_label['class_labels'] = torch.tensor(batch[0]["labels"])
        ret_label['image_id'] = torch.tensor(batch[0]["image_id"])
        ret_label['area'] = torch.tensor(batch[0]["area"])
        ret_label['iscrowd'] = torch.tensor(batch[0]["iscrowd"])
        
        pixel_values = [item["image"] for item in batch]
        encoded_input = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]
        batch = {}
        batch['pixel_values'] = encoded_input['pixel_values']
        batch['pixel_mask'] = encoded_input['pixel_mask']
        batch['labels'] = labels

        return batch, ret_label

if __name__ == '__main__':
    pass
