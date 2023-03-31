import os
import cv2
import torchvision.transforms as tfs
import torch.utils.data
import numpy as np
from skimage import io
import pickle

def get_train_val_dataloaders(cfgs):
    batch_size = cfgs.get('batch_size', 8)
    num_workers = cfgs.get('num_workers', 4)
    image_size = cfgs.get('image_size', 256)
    crop = cfgs.get('crop', None)
    use_mask = cfgs.get('use_mask', False)
    normalize = cfgs.get('normalize', True)


    data_dir = cfgs.get('data_dir', 'data/gbh/')
    data_split_dirs = cfgs.get('data_split_dirs', "data/split1+/")
    if isinstance(data_split_dirs, str):
        data_split_dirs = [data_split_dirs]

    data_train = {os.path.basename(data_split_dir): os.path.join(data_split_dir, 'train.txt') for data_split_dir in data_split_dirs}
    data_val = {os.path.basename(data_split_dir): os.path.join(data_split_dir, 'val.txt') for data_split_dir in data_split_dirs}
    image_paths = get_image_list(data_dir, list(data_train.values())+list(data_val.values()), False, use_vis)
    if not os.path.exists(os.path.join(data_dir, 'image_stats.pickle')):
        online_get_image_stats(data_dir, image_paths)

    ndsm_paths = get_image_list(data_dir, list(data_train.values())+list(data_val.values()), True)
    if not os.path.exists(os.path.join(data_dir, 'ndsm_stats.pickle')):
        online_get_ndsm_stats(data_dir, ndsm_paths)
    overfit = cfgs.get('overfit', False)

    train_loader = val_loader = None
    get_loader = lambda **kargs: get_tri_image_loader(**kargs, data_dir=data_dir, use_mask=use_mask, num_workers=num_workers, image_size=image_size, crop=crop, normalize=normalize, overfit=overfit)

    assert os.path.isdir(data_dir), "Data directory does not exist: %s" %data_dir
    print(f"Loading training data from {data_train}")
    train_loader = get_loader(data_split=data_train, is_validation=False, batch_size=batch_size)
    print(f"Loading validation data from {data_val}")
    if overfit:
        val_loader = train_loader
    else:
        val_loader = get_loader(data_split=data_val, is_validation=True, batch_size=batch_size)

    return train_loader, val_loader

def get_test_dataloaders(cfgs):
    num_workers = cfgs.get('num_workers', 4)
    image_size = cfgs.get('image_size', 256)
    crop = cfgs.get('crop', None)
    normalize = cfgs.get('normalize', True)
    use_mask = cfgs.get('test_use_mask', True)

    data_dir = cfgs.get('data_dir', 'data/gbh/')
    data_split_dirs = cfgs.get('test_data_split_dirs', ["data/SAO+", "data/MUC+", "data/GUA+", "data/split1+"])
    if isinstance(data_split_dirs, str):
        data_split_dirs = [data_split_dirs]
    data_test = {os.path.basename(data_split_dir): os.path.join(data_split_dir, 'test.txt') for data_split_dir in data_split_dirs}

    test_loader = {}
    get_loader = lambda **kargs: get_tri_image_loader(**kargs, data_dir=data_dir, use_mask=use_mask, num_workers=num_workers, image_size=image_size, crop=crop, normalize=normalize)

    assert os.path.isdir(data_dir), "Data directory does not exist: %s" %data_dir
    print(f"Loading testing data from {data_test}")
    for name, data in data_test.items():
        test_loader.update({name: get_loader(data_split={name: data}, is_validation=True, batch_size=1)})

    return test_loader

def compute_stats_per_image(mask, ndsm, res):
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_8U)
    num_instances = connected_components[0]
    instances = connected_components[1]

    for i in range(num_instances):
        m = (instances == i+1).astype(np.uint8)
        if m.max()<1:
            continue
        h_gt = np.median(ndsm[m>0])
        size = m.sum()
        if size in res:
            res[size].append(h_gt)
        else:
            res.update({size: [h_gt]})
    return res

def online_get_image_stats(data_dir, image_paths):
    sum_x = np.zeros(3)
    sum_x2 = np.zeros(3)
    sum_size = 0
    for image_path in image_paths:
        img = io.imread(image_path).astype(int).reshape(-1,3)
        sum_x += img.sum(axis=0)
        sum_x2 += (img*img).sum(axis=0)
        sum_size += img[:, 0].size
    mean = sum_x / sum_size
    std = list(np.sqrt(sum_x2/sum_size - mean*mean))
    mean = list(mean)
    torch.save([mean, std], os.path.join(data_dir, 'image_stats.pickle'))
    return mean, std

def online_get_ndsm_stats(data_dir, image_paths):
    sum_x = 0
    sum_x2 = 0
    sum_size = 0
    minh = 0
    maxh = 0
    for image_path in image_paths:
        img = io.imread(image_path).flatten()
        img = np.nan_to_num(img)
        sum_x += img.sum()
        sum_x2 += (img**2).sum()
        sum_size += img.size
        minh = img.min() if img.min()<minh else minh
        maxh = img.max() if img.max()>maxh else maxh
    minh = 0 if minh < 0 else minh
    mean = sum_x / sum_size
    std = np.sqrt(sum_x2/sum_size - mean**2)
    
    count = np.zeros(int(np.ceil(maxh)))
    for image_path in image_paths:
        img = io.imread(image_path).flatten()
        img = np.nan_to_num(img)
        img = np.clip(np.floor(img), a_min=0, a_max=None).astype(np.int)
        res = np.bincount(img)
        count[:res.size] += res
        
    torch.save([mean, std, minh, maxh, count], os.path.join(data_dir, 'ndsm_stats.pickle'))
    return mean, std, minh, maxh, count

class GBHDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_mode, image_size=256, use_mask=False, crop=None, hflip=False, \
        normalize=True, is_validation=False, overfit=False):
        super(GBHDataset, self).__init__()
        self.root = data_dir
        self.paths = make_gbh_dataset(data_dir, data_mode, overfit)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.hflip = hflip
        self.normalize = normalize
        self.use_mask = use_mask
        self.is_validation = is_validation
        image_stats_file = os.path.join(data_dir, 'image_stats.pickle')
        self.mean, self.std = torch.load(image_stats_file)

    def transform_ndsm(self, img):
        img_shifted = np.log(img - self.ndsm_min + 1)
        cls = np.floor(img_shifted / self.slope * self.num_classes)
        return cls

    def get_normal(self, ndsm):
        zx = cv2.Sobel(ndsm, cv2.CV_32F, 1, 0, ksize=5)
        zy = cv2.Sobel(ndsm, cv2.CV_32F, 0, 1, ksize=5)
        norm_tile = np.stack((-zx, -zy, np.ones_like(ndsm)))
        n = np.linalg.norm(norm_tile, axis=0)
        norm_tile /= n
        return norm_tile

    def transform(self, img):
        if self.crop:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        if not self.rcnn:
            img = tfs.functional.resize(img, (self.image_size, self.image_size))
        if self.hflip:
            img = tfs.functional.hflip(img)
        if self.normalize:
            img = tfs.functional.normalize(img, self.mean, self.std)
        return img

    def __getitem__(self, index):
        image_path, ndsm_path, mask_path = self.paths[index]
        file_idx = os.path.basename(image_path).split('_IMG')[0]
        image = torch.tensor(io.imread(image_path).astype(np.float32).transpose(2, 0, 1))
        ndsm = np.nan_to_num(np.float32(io.imread(ndsm_path))).clip(0)
        gt_dict = {"ndsm": torch.tensor(ndsm)[None, :, :]}
        if self.use_mask:
            mask = np.float32(io.imread(mask_path)>0)
            gt_dict.update({"mask": torch.tensor(mask)[None, :, :]})
        return file_idx, self.transform(image), gt_dict

    def __len__(self):
        return self.size

    def name(self):
        return 'GBHDataset'

## multiple image dataset with every subfolder in the data_dir ##
def make_gbh_dataset(dir, mode, overfit=False):
    folder_name = ['image', 'ndsm', 'mask']
    file_suffix = ['_IMG.tif', '_AGL.tif', '_BLG.tif']

    images = []
    with open(mode, 'r') as f:
        lines = f.readlines()
        if overfit:
            lines = lines[:2]
        for line in lines:
            image = []
            line = line.rstrip()
            for folder, suffix in zip(folder_name, file_suffix):
                image.append(os.path.join(dir, folder, line+suffix))
            images.append(image)
    return images

def get_image_list(dir, mode, ndsm=False, vis=False):
    images = []
    for m in mode:
        with open(m, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if ndsm:
                    images.append(os.path.join(dir, 'ndsm', line+'_AGL.tif'))
                elif vis:
                    images.append(os.path.join(dir, 'visualization', line+'_VIS.tif'))
                else:
                    images.append(os.path.join(dir, 'image', line+'_IMG.tif'))
    return images


def get_tri_image_loader(data_dir, data_split, is_validation=False, batch_size=8, num_workers=4, \
    image_size=256, crop=None, normalize=True, use_mask=False, use_normal=False, overfit=False, \
    rcnn=False, multitask=False, collate_fn=None, ordinal=False, num_classes=1000, sup_mode=None,
    tasks=[True, False], use_vis=False, extended=False, noise=False, prototype=False, drop_last=False):

    datasets = []
    dataset_class = GBHDataset
    if sup_mode is None:
        sup_mode = [False] * len(data_split)
    for ds, sm, task in zip(data_split.values(), sup_mode, tasks):
        datasets.append(dataset_class(data_dir, ds, image_size=image_size, crop=crop, normalize=normalize, \
        use_mask=use_mask, use_normal=use_normal, is_validation=is_validation, overfit=overfit, rcnn=rcnn, \
        multitask=multitask, ordinal=ordinal, num_classes=num_classes, sup_mode=sm, is_h=task, use_vis=use_vis, \
        noise=noise, prototype=prototype))
    dataset = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=drop_last
    )
    return loader