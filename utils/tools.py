import numpy as np
import cv2
from torchvision import transforms


def keypoint_painter(images, maps, img_h, img_w, numpy_array=False,
                     phase_gt=False, center_map=None):
    images = images.clone().cpu().data.numpy().transpose([0, 2, 3, 1])
    maps = maps.clone().cpu().data.numpy()
    if center_map is not None:
        center_map = center_map.clone().cpu().data.numpy()
    imgs_tensor = []
    if phase_gt:
        for img, map, c_map in zip(images, maps, center_map):
            img = cv2.resize(img, (img_w, img_h))
            for m in map[:14]:
                h, w = np.unravel_index(m.argmax(), m.shape)
                x = int(w * img_w / m.shape[1])
                y = int(h * img_h / m.shape[0])
                img = cv2.circle(img.copy(), (x, y), radius=1, thickness=2, color=(255, 0, 0))
            h, w = np.unravel_index(c_map.argmax(), c_map.shape)
            x = int(w * img_w / c_map.shape[1])
            y = int(h * img_h / c_map.shape[0])
            img = cv2.circle(img.copy(), (x, y), radius=1, thickness=2, color=(0, 0, 255))
            if numpy_array:
                imgs_tensor.append(img.astype(np.uint8))
            else:
                imgs_tensor.append(transforms.ToTensor()(img))
    else:

        for img, map_6 in zip(images, maps):
            img = cv2.resize(img, (img_w, img_h))
            for step_map in map_6:
                img_copy = img.copy()
                for m in step_map[:14]:
                    h, w = np.unravel_index(m.argmax(), m.shape)
                    x = int(w * img_w / m.shape[1])
                    y = int(h * img_h / m.shape[0])
                    img_copy = cv2.circle(img_copy.copy(), (x, y), radius=1, thickness=2, color=(255, 0, 0))
                if numpy_array:
                    imgs_tensor.append(img_copy.astype(np.uint8))
                else:
                    imgs_tensor.append(transforms.ToTensor()(img_copy))
    return imgs_tensor
