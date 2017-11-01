import os
import scipy.io
import numpy as np
import skimage.transform
import glob
import torch
from torch.utils.data import Dataset
import scipy.misc
import matplotlib.colors
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from torchvision import transforms
from PIL import Image

lms = None
imagefiles = None
weight = None

class LSPDataset(Dataset):
    def __init__(self, root_dir, transform=None, phase_train=True,
                 weighted_loss=False, bandwidth=50):
        self.scaled_h = 368
        self.scaled_w = 368
        self.map_h = 45
        self.map_w = 45
        self.guassian_sigma = 21
        self.num_keypoints = 14
        self.num_train = 11000
        global lms, imagefiles, weight
        if lms is None or imagefiles is None or weight is None:
            mat_lsp = scipy.io.loadmat(os.path.join(root_dir, 'lsp_dataset/joints.mat'),
                                       squeeze_me=True, struct_as_record=False)['joints']
            mat_lspet = scipy.io.loadmat(os.path.join(root_dir, 'lspet_dataset/joints.mat'),
                                         squeeze_me=True, struct_as_record=False)['joints']
            image_lsp = np.array(glob.glob(os.path.join(root_dir,
                                                        'lsp_dataset/images/*.jpg'), recursive=True))
            image_lspet = np.array(glob.glob(os.path.join(root_dir,
                                                          'lspet_dataset/images/*.jpg'), recursive=True))
            image_nums_lsp = np.array([float(s.rsplit('/')[-1][2:-4]) for s in image_lsp])
            image_nums_lspet = np.array([float(s.rsplit('/')[-1][2:-4]) for s in image_lspet])
            sorted_image_lsp = image_lsp[np.argsort(image_nums_lsp)]
            sorted_image_lspet = image_lspet[np.argsort(image_nums_lspet)]

            self.lms = np.append(mat_lspet.transpose([2, 1, 0])[:, :2, :],
                                 # only the x, y coords, not the "block or not" channel
                                 mat_lsp.transpose([2, 0, 1])[:, :2, :],
                                 axis=0)
            self.imagefiles = np.append(sorted_image_lspet, sorted_image_lsp)
            imgs_shape = []
            for img_file in self.imagefiles:
                imgs_shape.append(Image.open(img_file).size)
            lms_scaled = self.lms / np.array(imgs_shape)[:, :, np.newaxis]
            self.weight = np.logical_and(lms_scaled > 0, lms_scaled <= 1).astype(np.float32)
            self.weight = self.weight[:, 0, :] * self.weight[:, 1, :]
            self.weight = np.append(self.weight, np.ones((self.weight.shape[0], 1)), axis=1)
            self.weight = self.weight[:, np.newaxis, :].repeat(6, 1)
            if weighted_loss and phase_train:
                datas = lms_scaled[:self.num_train].reshape(self.num_train, -1)
                datas[datas < 0] = 0
                datas[datas > 1] = 0
                datas_pca = PCA(n_components=3).fit_transform(datas)
                kde = KernelDensity(bandwidth=bandwidth).fit(datas_pca)
                p = np.exp(kde.score_samples(datas_pca))
                p_median = np.median(p)
                p_weighted = p_median / p
                self.weight[:self.num_train] *= p_weighted[:, np.newaxis, np.newaxis]
            lms = self.lms
            imagefiles = self.imagefiles
            weight = self.weight
        else:
            self.lms = lms
            self.imagefiles = imagefiles
            self.weight = weight

        self.transform = transform
        self.phase_train = phase_train

    def __len__(self):
        if self.phase_train:
            return self.num_train
        else:
            return self.imagefiles.shape[0] - self.num_train

    def __getitem__(self, idx):
        if not self.phase_train:
            idx += self.num_train
        image = scipy.misc.imread(self.imagefiles[idx])
        image_h, image_w = image.shape[:2]
        lm = self.lms[idx].copy()
        lm[0] = lm[0] * self.map_w / image_w
        lm[1] = lm[1] * self.map_h / image_h
        gt_map = []
        for (x, y) in zip(lm[0], lm[1]):
            if x > 0 and y > 0:
                heat_map = guassian_kernel(self.map_w, self.map_h,
                                           x, y, self.guassian_sigma)
            else:
                heat_map = np.zeros((self.map_h, self.map_w))
            gt_map.append(heat_map)
        gt_map = np.array(gt_map)
        gt_backg = np.ones([self.map_h, self.map_w]) - np.max(gt_map, 0)
        gt_map = np.append(gt_map, gt_backg[np.newaxis, :, :], axis=0)

        center_x = (self.lms[idx][0][self.lms[idx][0] < image_w].max() +
                    self.lms[idx][0][self.lms[idx][0] > 0].min()) / 2
        center_y = (self.lms[idx][1][self.lms[idx][1] < image_h].max() +
                    self.lms[idx][1][self.lms[idx][1] > 0].min()) / 2

        center_x = center_x / image_w * self.scaled_w
        center_y = center_y / image_h * self.scaled_h

        center_map = guassian_kernel(self.scaled_w, self.scaled_h,
                                     center_x, center_y, self.guassian_sigma)
        weight = self.weight[idx]

        sample = {'image': image, 'gt_map': gt_map, 'center_map': center_map, 'weight': weight}

        if self.transform:
            sample = self.transform(sample)

        return sample


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new.astype(np.float32), 'gt_map': sample['gt_map'],
                'center_map': sample['center_map'], 'weight': sample['weight']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, phase_cuda=False):
        self.phase_cuda = phase_cuda

    def __call__(self, sample):
        image, gt_map, center_map, weight = sample['image'], sample['gt_map'], \
                                            sample['center_map'], sample['weight']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'gt_map': torch.from_numpy(gt_map).float(),
                'center_map': torch.from_numpy(center_map).float(),
                'weight': torch.from_numpy(weight).float()}


class Scale(object):
    def __init__(self, height, weight):
        self.height = height
        self.width = weight

    def __call__(self, sample):
        image, gt_map, center_map, weight = sample['image'], sample['gt_map'], \
                                            sample['center_map'], sample['weight']

        image = skimage.transform.resize(image, (self.height, self.width), preserve_range=True)
        return {'image': image, 'gt_map': sample['gt_map'],
                'center_map': sample['center_map'], 'weight': sample['weight']}
