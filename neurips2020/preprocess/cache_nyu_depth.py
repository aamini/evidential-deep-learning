import h5py
import os
from scipy.io import loadmat
import cv2
from tqdm import tqdm
import numpy as np

nyu_root = "/data/nyu_depth_processed"
train_path = "./data/depth_train_big_new.h5"
test_path = "./data/depth_test_big_new.h5"

if not path.isdir(nyu_root):
    raise ValueError("Please download the Apolloscape dataset to /data/apollo. \
        Or follow the instructions in the README to download a subset of the \
        data to test this the code provided in this repository.")


datasets = ["bedroom", "bathroom","cafe","bookstore","classroom","computer_lab","conference_room","dentist_office","dining_room","excercise_room","foyer","home_office","kitchen","library","living_room","office_","study_"]

scan_paths = []
dirs = set()

for dirName, subdirList, fileList in os.walk(nyu_root):
    if not any([d in dirName for d in datasets]):
        continue

    for fname in fileList:
        if "scan_" in fname:
            dirs.add(os.path.basename(dirName))
            scan_paths.append(os.path.join(dirName, fname))

print("found {} scans in {}".format(len(scan_paths), datasets))

imgs = []
depths = []
sorted(scan_paths)
sz = (160, 128)
for scan_path in tqdm(scan_paths):
    try:
        f = loadmat(scan_path)
    except:
        print("failed to load: {}".format(scan_path))
        continue

    img = f['rgb']
    depth = f['disp']
    imgs.append(cv2.resize(img, sz))
    depths.append(np.expand_dims(cv2.resize(depth, sz), -1))
    # if len(imgs)>100: break

print("saving")
n=len(imgs)
all_idx = np.arange(n)
train_idx = np.random.choice(all_idx, int(n*0.9),replace=False)
train_idx = sorted(train_idx)
test_idx = list(set(all_idx)-set(train_idx))
test_idx = sorted(test_idx)

imgs = np.array(imgs)
depths = np.array(depths)

f = h5py.File(train_path, 'w')
f.create_dataset("image", data=imgs[train_idx], dtype=np.uint8)
f.create_dataset("depth", data=depths[train_idx], dtype=np.uint8)
f.close()

f = h5py.File(test_path, 'w')
f.create_dataset("image", data=imgs[test_idx], dtype=np.uint8)
f.create_dataset("depth", data=depths[test_idx], dtype=np.uint8)
f.close()

import pdb; pdb.set_trace()
