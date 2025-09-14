from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, "rb") as img_file:
          magic_num, img_num, row, col = struct.unpack(">4i", img_file.read(16))
          assert(magic_num == 2051)
          tot_pixels = row * col
          X = np.vstack([np.array(struct.unpack(f"{tot_pixels}B", img_file.read(tot_pixels)), dtype=np.float32) for _ in range(img_num)])
          X -= np.min(X)
          X /= np.max(X)
        self.img = X

        with gzip.open(label_filename, "rb") as label_file:
          magic_num, label_num = struct.unpack(">2i", label_file.read(8))
          assert(magic_num == 2049)
          y = np.array(struct.unpack(f"{label_num}B", label_file.read()), dtype=np.uint8)
        self.label = y
        self.transforms = transforms
          ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X,y = self.img[index],self.label[index]
        if self.transforms:
          X = X.reshape((28,28,1))
          X = self.apply_transforms(X)
          return X.reshape((1,28*28)),y
        return X,y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.label.shape[0]
        ### END YOUR SOLUTION