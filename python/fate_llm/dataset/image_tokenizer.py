import os
import numpy as np
import pandas as pd
from federatedml.nn.dataset.base import Dataset
from federatedml.util import LOGGER
from federatedml.nn.dataset.image import ImageDataset


class TokenizerImageDataset(Dataset):

    """

    """

    def __init__(
            self,
            center_crop=False,
            center_crop_shape=None,
            generate_id_from_file_name=True,
            file_suffix='.png',
            float64=False,
            label_dtype='long'):

        super(TokenizerImageDataset, self).__init__()

        self.dataset = None
        self.center_crop = center_crop
        self.size = center_crop_shape
        self.generate_id_from_file_name = generate_id_from_file_name
        self.file_suffix = file_suffix
        self.float64 = float64
        self.label_type = label_dtype

    #Need to edit. Also need to figure out how to incorporate collate_fn
    def __getitem__(self, item):

        if item < 0:
            item = len(self) + item
        if item < 0:
            raise IndexError('index out of range')

        return item

    def __len__(self):
        len_ = 0
        if self.dataset is not None:
            len_ += len(self.dataset)
        return len_

    def load(self, file_path):

        # load dataset
        self.dataset = ImageDataset(
            center_crop=self.center_crop,
            center_crop_shape=self.size,
            generate_id_from_file_name=self.generate_id_from_file_name,
            file_suffix=self.file_suffix,
            float64=self.float64,
            label_dtype=self.label_type
        )
        if os.path.exists(file_path):
            self.dataset.load(file_path)
        else:
            self.dataset = None
            LOGGER.info(
                f' dataset not found in {file_path}, will not load dataset')
    
    def get_dataset(self):
        return self.dataset

    def get_classes(self):
        return self.dataset.get_classes()
