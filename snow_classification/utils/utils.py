import random

import cv2
import torch
import numpy as np


def grid_spread(kps, output_size, img_x, img_y, grid_size=10):
    if len(kps) <= output_size:
        return kps
    grid = [[[] for j in range(img_y//grid_size)] for i in range(img_x//grid_size)]
    for pt in kps:
        x = round(grid_size*pt.pt[0] // img_x)
        y = round(grid_size*pt.pt[1] // img_y)
        grid[x][y].append(pt)
    out = []
    grids = [(x,y) for x in range(grid_size) for y in range(grid_size)]
    while len(out) < output_size and len(grids) > 0:
        idx = random.randint(0, len(grids)-1)
        x, y = grids[idx]
        if len(grid[x][y]) > 0:
            pt = random.choice(grid[x][y])
            out.append(pt)
            grid[x][y].remove(pt)
        else:
            grids.pop(idx)
    return out


def ransac_metric(path: str, classifier, nfeatures: int = 500) -> float:
    torch.no_grad()

    FLANN_INDEX_KDTREE = 6
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    vid = cv2.VideoCapture(path)
    ret, frame = vid.read()
    orb = cv2.ORB_create(nfeatures=nfeatures)

    kp = orb.detect(frame, None)

    prev_good, _ = classifier.predict(kp, frame)
    _, prev_des = orb.compute(frame, prev_good)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    avg = 0
    avg2 = 0
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print(num_frames)

    ret, frame = vid.read()
    while ret:
        kp = orb.detect(frame, None)

        good, bad = classifier.predict(kp, frame)

        _, good_des = orb.compute(frame, good)
        matches = flann.knnMatch(good_des, prev_des, k=2)

        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for i, match in enumerate(matches):
            if len(match) < 2:
                continue
            m, n = match
            if m.distance < 0.8 * n.distance:
                pts2.append(prev_good[m.trainIdx].pt)
                pts1.append(good[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC)
        if F is not None:
            prev_good = good
            prev_des = good_des
            # We select only inlier points
            avg += len(pts1) - len(pts1[mask.ravel() == 1])
            avg2 += len(pts1[mask.ravel() == 1]) / len(pts1)
        ret, frame = vid.read()

    del vid

    print(avg / num_frames)
    print(avg2 / num_frames)
    return avg / num_frames


def get_patch(padded_img, pt, scales=None, patch_size=60):
    if scales is None:
        scales = [1]
    offset = patch_size // 2
    patches = []
    for i, scale in enumerate(scales):
        scaled_patch_size = int(patch_size * scale)
        patch = padded_img[pt[1] + offset - scaled_patch_size // 2:pt[1] + offset + scaled_patch_size // 2,
                pt[0] + offset - scaled_patch_size // 2:pt[0] + offset + scaled_patch_size // 2]
        patches.append(cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR) if scaled_patch_size != patch_size else patch)
    return np.stack(patches, axis=0).astype(np.single)

def pad_image(img, padding):
    return np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='median')

def get_patches(padded_img, pts, scales=None, patch_size=60):
    if scales is None:
        scales = [1]

    scaled_patches = []
    indices = np.transpose(pts)
    for i, scale in enumerate(scales):
        m, n = padded_img.shape
        K = int(scale * patch_size // 2)
        R = np.arange(-K, K + 1)
        patches = np.take(padded_img, R[:, None] * n + R + (indices[0] * n + indices[1])[:, None, None])

        scaled_patches.append(patches)


    return scaled_patches


def to_binary_vector(descriptors):
    out = []
    if descriptors is None:
        return out
    for d in descriptors:
        out.append(((d[:, None] & (1 << np.arange(8))) > 0).flatten().astype(np.single))
    return out


def strided_extract_snow_mask(snow, patch_size=60, stride=10):
    snow_mask = np.zeros(shape=snow.shape[:2], dtype=np.single)
    mask_count = np.zeros_like(snow_mask)

    if stride is None:
        stride = patch_size

    minimum_distance = 15
    snow_threshhold = 20

    for y in range(0, snow.shape[0] - patch_size + stride, stride):
        for x in range(0, snow.shape[1] - patch_size + stride, stride):

            snow_patch = snow[y: y + patch_size, x: x + patch_size]

            patch_median = np.median(snow_patch, axis=(0, 1))

            patch_mask = np.linalg.norm(np.subtract(snow_patch, patch_median, dtype=np.single), axis=2)

            patch_mask[patch_mask < minimum_distance] = 0
            if np.max(patch_mask) > 0:
                patch_mask /= np.max(patch_mask)

            snow_mask[y: y + patch_size, x: x + patch_size] += patch_mask
            mask_count[y: y + patch_size, x: x + patch_size] += np.ones_like(patch_mask)

    snow_mask = np.divide(snow_mask, mask_count)

    bw_snow = cv2.cvtColor(snow, cv2.COLOR_BGR2GRAY)
    if snow_threshhold:
        snow_mask[bw_snow < snow_threshhold] /= 3

    bw_snow = cv2.cvtColor(bw_snow, cv2.COLOR_GRAY2BGR)
    bw_snow = np.multiply(bw_snow, snow_mask[:, :, None])

    superimposed_snow = np.multiply(snow, snow_mask[:, :, None])
    return superimposed_snow, bw_snow, snow_mask


def extract_snow_mask(snow, patch_size=40):
    snow_mask = np.zeros(shape=snow.shape[:2], dtype=np.single)
    superimposed_snow = np.zeros_like(snow, dtype=np.single)
    bw_snow = np.zeros_like(snow, dtype=np.single)
    stride = patch_size

    minimum_distance = 30
    snow_threshhold = 20

    for y in range(0, snow.shape[0] - patch_size, stride):
        for x in range(0, snow.shape[1] - patch_size, stride):

            snow_patch = snow[y: y + patch_size, x: x + patch_size]

            patch_median = np.median(snow_patch, axis=(0, 1))

            patch_mask = np.linalg.norm(np.subtract(snow_patch, patch_median, dtype=np.single), axis=2)

            patch_mask[patch_mask < minimum_distance] = 0
            if np.max(patch_mask) > 0:
                patch_mask /= np.max(patch_mask)
            bw_snow_patch = cv2.cvtColor(snow_patch, cv2.COLOR_BGR2GRAY)

            if snow_threshhold:
                patch_mask[bw_snow_patch < snow_threshhold] /= 3

            bw_snow_patch = cv2.cvtColor(bw_snow_patch, cv2.COLOR_GRAY2BGR)

            snow_patch = np.multiply(snow_patch, patch_mask[:, :, None])
            snow_mask[y: y + patch_size, x: x + patch_size] = patch_mask
            superimposed_snow[y: y + patch_size, x: x + patch_size] = snow_patch
            bw_snow[y: y + patch_size, x: x + patch_size] = np.multiply(bw_snow_patch, patch_mask[:, :, None])

    return superimposed_snow, bw_snow, snow_mask


def superimpose_snow(snow, snow_mask, background):
    temp_background = np.multiply(background.astype(np.single), (1 - snow_mask[:, :, None]))
    out = temp_background + snow

    return out


if __name__ == "__main__":
    grid_spread(None, 100, 1920, 1080)
    quit(0)
    path = "noaa_snow.mov"

    vid = cv2.VideoCapture(path)
    vid2 = cv2.VideoCapture("demo1.mov")

    if vid.isOpened() and vid2.isOpened():
        ret, frame = vid.read()
        ret2, frame2 = vid2.read()
        num = 0
        while ret and ret2:
            snow, bw_snow, mask = strided_extract_snow_mask(frame, stride=30, patch_size=60)
            frames = [superimpose_snow(bw_snow, mask, frame2), frame]
            cv2.imshow('Frame', frames[num % 2] / 255)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                num += 1
            ret, frame = vid.read()
            ret2, frame2 = vid2.read()
    vid.release()
    cv2.destroyAllWindows()
