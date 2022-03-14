from torch import from_numpy


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        patch, label = sample['sample'], sample['label']


        return {'sample': from_numpy(sample['sample']),
                'label': from_numpy(sample['label'])}



