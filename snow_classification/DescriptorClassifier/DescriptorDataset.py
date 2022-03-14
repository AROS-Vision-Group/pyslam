import math
import random

import numpy

import numpy as np
import glob
from torch.utils.data import Dataset
import sys
from tqdm import tqdm

class DescriptorDataset(Dataset):
    def __init__(self, path, transform=None, files_per_cache=None, verbose=False):
        super().__init__()

        self.transform = transform

        self.cache_idx = 0

        tp_paths = glob.glob(path + "tp/*.npy")
        fp_paths = glob.glob(path + "fp/*.npy")

        # Load all descriptors into lists
        self.descriptors = []
        self.descriptor_paths = []
        paths = tp_paths + fp_paths
        self.labels = []

        self.num_caches = 1
        if files_per_cache:
            self.num_caches = math.ceil(len(paths) / files_per_cache)
            if verbose:
                print(f"Using {self.num_caches} caches in cached dataset")


        self.size = 0
        self.get_data_size(paths)
        self.verbose = verbose

        self.descriptor_files_per_cache = len(paths) // self.num_caches
        self.descriptors_from_prev_batches = 0

        labels = [np.array([0], dtype=numpy.single) for i in range(len(tp_paths))] + [np.array([1], dtype=numpy.single)
                                                                                      for i in range(len(fp_paths))]

        p = np.random.permutation(len(paths))
        for i in p:
            self.descriptor_paths.append(paths[i])
            self.labels.append(labels[i])

        self.load_cache(self.descriptor_paths, self.labels)

    def __len__(self):
        return self.size

    def get_data_size(self, paths):
        for path in paths:
            descriptors = np.load(path, allow_pickle=True)
            if np.atleast_1d(descriptors).any():
                self.size += len(descriptors)

    def __getitem__(self, item):
        cache_max = self.descriptors_from_prev_batches + len(self.descriptors)

        if item >= cache_max:
            self.cache_idx = self.cache_idx + 1
            self.descriptors_from_prev_batches = len(self.descriptors) + self.descriptors_from_prev_batches
            self.load_cache(self.descriptor_paths, self.labels)

        desc, label = self.descriptors[item - self.descriptors_from_prev_batches]
        sample = {'sample': desc, 'label': label}

        if self.num_caches > 1 and item == self.size - 1:
            self.cache_idx = 0
            self.descriptors_from_prev_batches = 0
            self.load_cache(self.descriptor_paths, self.labels)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_cache(self, path_list, labels):
        if self.verbose:
            tqdm.write(f"Loading cache {self.cache_idx}")

        self.descriptors = []
        end = None if self.cache_idx == self.num_caches - 1 else (self.cache_idx + 1) * self.descriptor_files_per_cache
        for path, label in zip(path_list[self.cache_idx * self.descriptor_files_per_cache:end], labels[self.cache_idx * self.descriptor_files_per_cache:end]):
            descriptors = np.load(path, allow_pickle=True)
            if np.atleast_1d(descriptors).any():
                self.descriptors.extend([(desc, label) for desc in self.to_binary_vector(descriptors)])
        random.shuffle(self.descriptors)

        if self.verbose:
            tqdm.write(f"Finished loading cache {self.cache_idx} with size {len(self.descriptors)}")


    def to_binary_vector(self, descriptors):
        out = []
        for d in descriptors:
            out.append((((d[:,None] & (1 << np.arange(8)))) > 0).flatten().astype(np.single))
        return out


if __name__ == "__main__":
    test = DescriptorDataset("Data/train/")
    print(test.__len__())
    desc = test.__getitem__(0)
    print(desc)
