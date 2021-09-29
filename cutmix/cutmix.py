import numpy as np
import random
from torch.utils.data.dataset import Dataset

from cutmix.utils import rand_bbox

from imagecorruptions import corrupt
from PIL import Image

import torchvision.datasets as datasets

class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=1.0, transform=None):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.transform = transform

    def __getitem__(self, index):
        img, lb = self.dataset[index]

        # i have a transformed image which has fixed dimensions
        img = np.asarray(img)
        normal_img = Image.fromarray(np.copy(img))

        # generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = random.choice(range(len(self)))
        img2, lb2 = self.dataset[rand_index]
        while lb2 == lb:
            rand_index = (rand_index + 1) % len(self)
            img2, lb2 = self.dataset[rand_index]
        img2 = np.asarray(img2)
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape, lam)
        img[bbx1:bbx2, bby1:bby2, :] = img2[bbx1:bbx2, bby1:bby2, :]
        
        # corrupt image here
        corr_number = np.random.choice(15)
        severity_index = np.random.choice(5)
        corr_img = corrupt(img, severity=severity_index+1, corruption_number=corr_number)
        corr_img = Image.fromarray(corr_img)

        if self.transform:
            normal_img, corr_img = self.transform[0](normal_img), self.transform[1](corr_img)

        return normal_img, lb, corr_img, self.num_class

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    traindir = "/home/jatin/Imagenet/ILSVRC/Data/CLS-LOC/train"
    train_set = datasets.ImageFolder(traindir)

    cnc_set = CutMix(train_set, 1000)

    img, lb, corr, lb2 = cnc_set[256]

    corr.save("cnc.jpg")
    img.save("normal.jpg")
