import torch
import torchvision.transforms.functional as tf
import random

import cv2


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        patch, label = sample['sample'], sample['label']

        # swap color axis because
        # numpy image: S x H x W x C
        # torch image: S x C x H x W
        patch = patch.transpose((0, 3, 1, 2))/255.0
        return {'sample': torch.from_numpy(patch),
                'label': torch.from_numpy(label)}


class Normalize(object):

    def __init__(self, mean, std, inplace):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        patch, label = sample['sample'], sample['label']

        return {'sample': tf.normalize(patch, self.mean, self.std, self.inplace),
                'label': label}


class ToNumpy(object):

    def __call__(self, sample):
        patch, label = sample['sample'], sample['label']

        # swap color axis because
        # numpy image: S x H x W x C
        # torch image: S x C x H x W
        patch = patch.transpose(3, 1)
        patch = patch.transpose(2, 1)
        return {'sample': patch.numpy(),
                'label': label.numpy()}


class RandomRotate(object):

    def __call__(self, sample):
        patch, label = sample['sample'], sample['label']
        angle = random.choice([90, 180, 270])
        for i, scale in enumerate(patch):
            patch[i] = tf.rotate(scale, angle)
        return {'sample': patch, 'label': label}


# if __name__ == "__main__":
#     test = PatchDataset("Data/train/")
#     patches = test.__getitem__(0)
#     rr = RandomRotate()
#     tt = ToTensor()
#     tn = ToNumpy()
#     patches = tt(patches)
#     patches = rr(patches)
#     patches = tn(patches)
#     cv2.imshow("Window", patches['sample'][0])
#     while True:
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             print('Quitting, \'q\' pressed.')
#             continue