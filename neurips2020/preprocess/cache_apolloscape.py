import h5py
import os
from scipy.io import loadmat
import cv2
from tqdm import tqdm
import numpy as np
import glob

root = "/data/apollo"
test_path = "/data/apolloscape_test.h5"
if not path.isdir(root):
    raise ValueError("Please download the Apolloscape dataset to /data/apollo. \
        Or follow the instructions in the README to download a subset of the \
        data to test this the code provided in this repository.")


img_paths = sorted(glob.glob(os.path.join(root, "camera_5/*.jpg")))
disp_paths = sorted(glob.glob(os.path.join(root, "disparity/*.png")))
inds = np.random.choice(len(img_paths), 1000, replace=False)


sz = (160, 128)
def resize(I):
    w_ = int(I.shape[1] / (float(I.shape[0]) / float(sz[1])))
    I = cv2.resize(I, (w_, sz[1]))
    I = I[:, 100:-100]
    I = cv2.resize(I, sz)
    return I

imgs = []
disps = []
for i, ind in enumerate(tqdm(inds)):
    img = cv2.imread(img_paths[ind])
    imgs.append(resize(img))

    disp = cv2.imread(disp_paths[ind], 0)
    disps.append(np.expand_dims(resize(disp), -1))


print("saving")

f = h5py.File(test_path, 'w')
f.create_dataset("image", data=imgs, dtype=np.uint8)
f.create_dataset("depth", data=disps, dtype=np.uint8)
f.close()

import pdb; pdb.set_trace()
