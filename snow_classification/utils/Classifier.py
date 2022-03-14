import numpy as np
import cv2
from snow_classification.PatchClassifier.transforms import ToTensor as PatchToTensor
import torch
from snow_classification.PatchClassifier.patchClassifier import PatchClassifier as pclf
from snow_classification.DescriptorClassifier.descriptorClassfier import DescriptorClassifier as dclf
from snow_classification.utils.utils import get_patch, to_binary_vector, pad_image


class Classifier:
    def predict(self, kp, img: np.ndarray) -> (list, list):
        pass


class PatchPredictor(Classifier):
    def __init__(self, device, scales=None, patch_size=64, state_dict=None, state_dict_path=None, batch_size=32):
        self.batch_size = batch_size
        self.scales = scales
        if scales is None:
            self.scales = [1, 0.75, 0.5]
        self.patch_size = patch_size
        self.classifier = pclf(patch_size=patch_size)
        self.device = device
        if state_dict_path is not None:
            self.classifier.load_state_dict(torch.load(state_dict_path))
        elif state_dict is not None:
            self.classifier.load_state_dict(state_dict)
        self.classifier.to(device)

    def predict(self, kp, img: np.ndarray) -> (list, list):
        patches = []
        to_tensor = PatchToTensor()
        pad_img = pad_image(img, self.patch_size//2)
        for pt in kp:
            patch = {'sample': get_patch(pad_img, (int(pt.pt[0]), int(pt.pt[1])), scales=self.scales, patch_size=self.patch_size),
                     'label': np.array([0])}
            patches.append(to_tensor(patch)['sample'])
        good = []
        bad = []
        for i in range(0, len(patches), self.batch_size):
            tensors = torch.stack(patches[i:i + self.batch_size])
            tensors = tensors.to(self.device)
            pred = self.classifier(tensors).cpu().flatten()
            for j, p in enumerate(pred):
                if p < 0.5:
                    good.append(kp[i+j])
                else:
                    bad.append(kp[i+j])
        return good, bad

    def predict_and_filter(self, kp, des, img: np.ndarray) -> (list, list):
        patches = []
        to_tensor = PatchToTensor()
        pad_img = pad_image(img, self.patch_size//2)
        for pt in kp:
            patch = {'sample': get_patch(pad_img, (int(pt.pt[0]), int(pt.pt[1])), scales=self.scales, patch_size=self.patch_size),
                     'label': np.array([0])}
            patches.append(to_tensor(patch)['sample'])
        good = []
        good_des = []
        for i in range(0, len(patches), self.batch_size):
            tensors = torch.stack(patches[i:i + self.batch_size])
            tensors = tensors.to(self.device)
            pred = self.classifier(tensors).cpu().flatten()
            for j, p in enumerate(pred):
                if p < 0.5:
                    good.append(kp[i+j])
                    good_des.append(des[i + j])

        return np.array(good), np.array(good_des)


class DescriptorPredictor(Classifier):
    def __init__(self, device, input_size=256, n_features=2000, state_dict=None, state_dict_path=None, batch_size=128):
        self.batch_size = batch_size
        self.classifier = dclf(input_size=input_size)
        if state_dict_path is not None:
            self.classifier.load_state_dict(torch.load(state_dict_path))
        elif state_dict is not None:
            self.classifier.load_state_dict(state_dict)
        self.orb = cv2.ORB_create(n_features)
        self.device = device
        self.classifier.to(device)

    def predict(self, kp, img: np.ndarray) -> (list, list):
        _, des = self.orb.compute(img, kp)
        des = to_binary_vector(des)
        des = torch.from_numpy(np.array(des))
        pred = self.classifier(des.to(self.device)).cpu().flatten()
        good = []
        bad = []
        for i, p in enumerate(pred):
            if p < 0.5:
                good.append(kp[i])
            else:
                bad.append(kp[i])
        return good, bad

    def predict_and_filter(self, kp, des, img: np.ndarray) -> (list, list):
        bin_des = to_binary_vector(des)
        bin_des = torch.from_numpy(np.array(bin_des))
        pred = self.classifier(bin_des.to(self.device)).cpu().flatten()
        good = []
        good_des = []
        for i, p in enumerate(pred):
            if p.item() < 0.5:
                good.append(kp[i])
                good_des.append(des[i])
        return good, np.array(good_des)


class AllGoodClassifier(Classifier):
    def predict(self, kp, img: np.ndarray) -> (list, list):
        return kp, []
