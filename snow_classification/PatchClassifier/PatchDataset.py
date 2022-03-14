import random

import numpy
from collections import defaultdict
import numpy as np
import glob
from torch.utils.data import Dataset
import cv2
from re import sub

from tqdm import tqdm

from utils.lru_list import LRUCache
from utils.utils import pad_image, get_patch


def discrete_trapezoid_pmf(min_height, points, slope):
    if slope == 0:
        return [1 / points for i in range(points)]

    b = min_height if slope > 0 else min_height - slope

    pmf = [slope * x / points + b for x in range(points)]
    pmf = np.array(pmf)

    return pmf / np.sum(pmf)


class PatchDataset(Dataset):
    def __init__(self, path, transform=None, scales=None, random_order=False, train=False):
        super().__init__()

        self.patch_size = 64
        if scales is None:
            self.scales = [1, 0.75, 0.5]
        else:
            self.scales = scales


        self.pmf_slope = -1.7
        self.pmf_min_height = 0.4

        self.cache_size = 250

        self.transform = transform

        if train:
            self.tp_folder = "tp_bps"
            self.fp_folder = "fp"
        else:
            self.tp_folder = "tp"
            self.fp_folder = "fp"
        tp_paths = glob.glob(path + self.tp_folder + "\\*.txt")
        fp_paths = glob.glob(path + self.fp_folder + "\\*.txt")
        self.img_folder = "imgs"
        self.shuffle_width = 100
        if random_order:
            self.shuffle_width = 100
            tp_in_last_window = random.sample(range(len(tp_paths)), self.shuffle_width // 2)
            fp_in_last_window = random.sample(range(len(fp_paths)), self.shuffle_width // 2)

            tp_in_last_window = [tp_paths.pop(i) for i in sorted(tp_in_last_window, reverse=True)]
            fp_in_last_window = [fp_paths.pop(i) for i in sorted(fp_in_last_window, reverse=True)]

        self.pts = []
        self.image_cache = LRUCache(self.cache_size)
        self.file_paths = []
        self.labels = []

        files = tp_paths + fp_paths
        labels = [np.array([0], dtype=numpy.single) for i in range(len(tp_paths))] + [np.array([1], dtype=numpy.single)
                                                                                      for i in range(len(fp_paths))]

        p = np.random.permutation(len(files))

        self.path_indeces = defaultdict(lambda: None)
        self.image_paths = []

        for i in p:
            self.file_paths.append(files[i])
            self.labels.append(labels[i])
        if random_order:
            for i in range(self.shuffle_width):
                files.append(tp_in_last_window[i // 2] if i % 2 == 0 else fp_in_last_window[(i - 1) % 2])
                labels.append([0] if i % 2 == 0 else [1])

        self.load_shuffled_points()


    def __len__(self):
        return len(self.pts)

    def __getitem__(self, item):
        path_index = self.pts[item]['img_index']
        label = self.pts[item]['label']
        img = self.image_cache.get(path_index)

        if img is None:
            img = cv2.imread(self.image_paths[path_index])
            img = pad_image(img, self.patch_size // 2)
            self.image_cache.put(path_index, img)

        patch = get_patch(img, self.pts[item]['pt'], patch_size=self.patch_size, scales=self.scales)
        sample = {'sample': patch, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_shuffled_points(self):
        file_list = self.file_paths[:self.shuffle_width]
        dist = discrete_trapezoid_pmf(self.pmf_min_height, self.shuffle_width, self.pmf_slope)

        next_file = self.shuffle_width
        remaining_points = {path: self.load_pts_and_image_path(path, i) for i, path in enumerate(file_list)}
        indeces = np.arange(self.shuffle_width)

        # subarray_lengths = defaultdict(lambda: 0)
        # lengths = []
        while len(file_list) > 0:
            p = dist if next_file < len(self.file_paths) else None
            file_index = np.random.choice(indeces if len(indeces) == len(file_list) else np.arange(len(file_list)), p=p)
            new_point = remaining_points[file_list[file_index]].pop() if len(
                remaining_points[file_list[file_index]]) > 0 else None
            # if subarray_lengths[file_list[file_index]] == 0:
            #    subarray_lengths[image_list[img_index]] = len(self.pts)
            if new_point is None:
                del remaining_points[file_list[file_index]]

                # lengths.append(len(self.pts) - subarray_lengths[file_list[file_index]])
                file_list.pop(file_index)
                if next_file < len(self.file_paths):
                    file_list.append(self.file_paths[next_file])
                    remaining_points[self.file_paths[next_file]] = self.load_pts_and_image_path(
                        self.file_paths[next_file], next_file)
                    next_file += 1

            else:
                self.pts.append(new_point)
        del self.path_indeces
        del self.labels
        del self.file_paths

        # lengths = np.array(lengths)
        # print(f"Max length {np.max(lengths)}, min {np.min(lengths)}, median {np.median(lengths)}, mean {np.mean(lengths)}, std {np.std(lengths)}")

    def load_pts_and_image_path(self, path, label_index):
        img_path = sub(rf"\\{self.tp_folder}\\|\\{self.fp_folder}\\", rf"\\{self.img_folder}\\", path)
        img_path = img_path.replace("_points.txt", ".png")

        if self.path_indeces[img_path] is None:
            self.path_indeces[img_path] = len(self.image_paths)
            img_index = len(self.image_paths)
            self.image_paths.append(img_path)
        else:
            img_index = self.path_indeces[img_path]

        with open(path, 'r') as f:
            pts = [tuple(map(int, line.split(" "))) for line in f.readlines()]
            # return [{'label': self.labels[label_index], "img_index": img_index, 'pt': pt} for pt in pts if pt[0] < self.patch_size // 2 or pt[1] < self.patch_size // 2 or pt[0] >= 1920 - self.patch_size // 2 or pt[1] >= 1080 - self.patch_size // 2]
            return [{'label': self.labels[label_index], "img_index": img_index, 'pt': pt} for pt in pts]

    def old_get_patch(self, img, pt, scales=None, patch_size=60):
        if scales is None:
            scales = [1]
        max_patch_size = int(np.ceil(patch_size * np.max(scales)))
        max_patch_size += max_patch_size % 2
        max_patch_size = patch_size
        padded_image = np.zeros((img.shape[0] + max_patch_size, img.shape[1] + max_patch_size, img.shape[2]),
                                dtype=numpy.single)
        padded_image[max_patch_size // 2: -max_patch_size // 2, max_patch_size // 2: -max_patch_size // 2] = img
        offset = max_patch_size // 2
        patches = np.zeros((len(scales), max_patch_size, max_patch_size, img.shape[2]), dtype=numpy.single)
        for i, scale in enumerate(scales):
            scaled_patch_size = int(patch_size * scale)
            patch = padded_image[pt[1] + offset - scaled_patch_size // 2:pt[1] + offset + scaled_patch_size // 2,
                    pt[0] + offset - scaled_patch_size // 2:pt[0] + offset + scaled_patch_size // 2]
            patches[i] = cv2.resize(patch, (max_patch_size, max_patch_size), interpolation=cv2.INTER_LINEAR)
        return patches


if __name__ == "__main__":
    maxs = []
    meds = []
    for j in tqdm(range(100)):
        test = PatchDataset(r"..\OrbDatasets\DarkDataset\points\\")
        print(test.__len__())
        patches = test.__getitem__(1)


        streak_label = patches["label"]
        streaks = []
        streak = 1
        for i in range(1, test.__len__()):
            patches = test.__getitem__(i)
            label = patches["label"]
            if streak_label == label:
                streak += 1
            else:
                streaks.append(streak)
                streak = 1
        maxi = np.max(streaks)
        med = np.mean(streaks)
        print("Max", maxi, "mean", med)
        maxs.append(maxi)
        meds.append(med)
    print(
        f"Max length {np.max(maxs)}, min {np.min(maxs)}, median {np.median(maxs)}, mean {np.mean(maxs)}, std {np.std(maxs)}")
    print(
        f"Max length {np.max(meds)}, min {np.min(meds)}, median {np.median(meds)}, mean {np.mean(meds)}, std {np.std(meds)}")
        #cv2.imshow("Window", patches['sample'][0] / 255)
        #print(patches["label"])
        #while True:
        #    key = cv2.waitKey(1) & 0xFF
        #    if key == ord('q'):
        #        print('Quitting, \'q\' pressed.')
        #        break

    # print(test.image_cache.hit_ratio())
    # img = cv2.imread("C:/Users/Eirik/Documents/NTNU/master/points/fp/eelume_2020-10-19_21_37_44_5.mp4_4.png")
    # patch_size = 24
    # img = np.zeros((501, 501, 3))
    # img[250, 250, :] = [255,255,255]
    # img = pad_image(img, patch_size //2)
    # patches = get_patch(img, (250, 250), patch_size=patch_size, scales=[1, 0.6, 0.3])
# print(patches[0])
