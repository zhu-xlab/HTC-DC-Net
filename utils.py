import os
from typing import Dict, List
import warnings
import yaml
#import zipfile
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
#import glob
import cv2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class UpdatableDict(Dict):
    def __init__(self):
        self.reset()

    def reset(self):
        self.dict = {}

    def update(self, val):
        for key, value in val.items():
            if key not in self.dict:
                self.dict.update({key: value})
            else:
                self.dict[key] += value
    
    def __call__(self):
        return self.dict

    def __getitem__(self, key):
        return self.dict[key]

    def __call__(self):
        return self.dict
    
    def __getitem__(self, key):
        return self.dict[key]

def load_yaml(path):
    print(f"Loading configs from {path}")
    with open(path, 'r') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # suppress the warning caused by yaml.safe_load, because we need objects in .yaml file to be read.
            try:
                return yaml.load(f)
            except:
                return yaml.load(f, Loader=yaml.FullLoader)

def xmkdir(path):
    """Create directory PATH recursively if it does not exist."""
    os.makedirs(path, exist_ok=True)

'''
def archive_code(arc_path, filetypes=['.py', '.yml']):
    print(f"Archiving code to {arc_path}")
    xmkdir(os.path.dirname(arc_path))
    zipf = zipfile.ZipFile(arc_path, 'w', zipfile.ZIP_DEFLATED)
    cur_dir = os.getcwd()
    flist = []
    for ftype in filetypes:
        flist.extend(glob.glob(os.path.join(cur_dir, '**', '*'+ftype), recursive=True))
    [zipf.write(f, arcname=f.replace(cur_dir,'archived_code', 1)) for f in flist]
    zipf.close()
'''
def save_config(cfgs, arc_path):
    print(f"Archiving configs to {arc_path}")
    xmkdir(os.path.dirname(arc_path))
    with open(arc_path, 'w') as f:
        yaml.dump(cfgs, f)

def fix_seed_for_reproducability(seed):
    """
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic. 

    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068 
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi 
    """
    # Use random seed.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def data_to_device(data, device='cuda'):
    if isinstance(data, int) | isinstance(data, float):
        return data
    elif isinstance(data, torch.Tensor):
        if device == 'cpu_test':
            if data.numel()>1:
                return data.to('cpu').numpy()
            else:
                return data.to('cpu').item()
        else:
            return data.to(device)
    elif isinstance(data, Dict):
        return {key: data_to_device(data[key], device) for key in data}
    elif isinstance(data, List):
        return [data_to_device(d, device) for d in data]
    else:
        NotImplementedError
'''
def find_latest_experiment(checkpoint_dir):
    latest_dir = sorted(os.listdir(checkpoint_dir))[-1]
    return os.path.join(checkpoint_dir, latest_dir)
'''

def convert_from_string(string):
    if string.lower() == 'true':
        return True
    elif string.lower() == 'false':
        return False
    try:
        return int(string)
    except:
        try:
            return float(string)
        except:
            return string

def save_intermediate_results_hook(savepath_prefix, image_id, save_format, save_input=False):
    def hook(module, input, output):
        if save_input:
            torch.save(input, savepath_prefix+image_id+save_format)
        else:
            torch.save(output, savepath_prefix+image_id+save_format)
    return hook



def compute_median(mask, pred, gt):
    mask = mask.cpu().numpy().astype(np.uint8)
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_8U)
    num_instances = connected_components[0]
    instances = connected_components[1]

    pred_l1 = torch.zeros_like(pred, device='cuda')
    gt_l1 = torch.zeros_like(pred, device='cuda')
    pred_h = []
    gt_h = []
    for i in range(num_instances):
        m = torch.tensor(instances == i+1, device='cuda')
        if m.max()<1:
            continue
        h_gt = torch.median(gt[m>0])
        h_pred = torch.median(pred[m>0])
        pred_l1 += (h_pred * m)
        gt_l1 += (h_gt * m)
        pred_h.append(h_pred)
        gt_h.append(h_gt)
    if len(pred_h) == 0:
        return pred_l1, gt_l1, torch.zeros(0).to(pred_l1.device), torch.zeros(0).to(pred_l1.device)
    else:
        return pred_l1, gt_l1, torch.stack(pred_h), torch.stack(gt_h)
