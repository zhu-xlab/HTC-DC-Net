import torch
from torch import nn
import torch.nn.functional as F
import wandb
from skimage import io
import os

from utils import compute_median

class BaseHeightPredictor(nn.Module):
    def __init__(self, cfgs=None):
        super(BaseHeightPredictor, self).__init__()
        self.test = cfgs.get("test", False)
        self.stats_file = os.path.join('data/split1+/stats.pkl')
    
    @staticmethod
    def _get_losses(cls, pred, gt):
        flag = gt["sup_mode"][:, None, None, None]
        if "mask" in gt:
            mask = gt["mask"] * flag + (~flag)
        else:
            mask = gt["ndsm"]>0
        losses = {
            "mae": (F.l1_loss(pred["ndsm"], gt["ndsm"], reduction='none') * mask).sum() / mask.sum()
                }
        losses.update({"loss_total": sum(losses.values())})
        return losses

    def get_losses(self, pred, gt):
        return self._get_losses(self, pred, gt)

    @staticmethod
    def _get_metric_params(cls, pred, gt, num_bin=101, interval=5):
        stats = torch.load(cls.stats_file).long().to(pred["ndsm"].device)
        cls.num_cls = stats.max() + 1
        metric_params = {
            "sum": pred["ndsm"].numel(),
            "ae": torch.abs(gt["ndsm"] - pred["ndsm"]).sum(),
            "se": ((gt["ndsm"] - pred["ndsm"])**2).sum()
        }
        gt_int = torch.clamp(gt["ndsm"], min=0, max=398).long()
        gt_cls = stats[gt_int].squeeze(1)
        gt_cls = F.one_hot(gt_cls, num_classes=cls.num_cls).permute(0, 3, 1, 2).contiguous()
        sum_cls = gt_cls.sum(dim=-1).sum(dim=-1).sum(dim=0)
        ae_cls = (torch.abs(gt["ndsm"] - pred["ndsm"]) * gt_cls).sum(dim=-1).sum(dim=-1).sum(dim=0)
        se_cls = ((gt["ndsm"] - pred["ndsm"])**2 *gt_cls).sum(dim=-1).sum(dim=-1).sum(dim=0)
        metric_params.update({"sum"+str(i): sum_cls[i] for i in range(cls.num_cls)})
        metric_params.update({"ae"+str(i): ae_cls[i] for i in range(cls.num_cls)})
        metric_params.update({"se"+str(i): se_cls[i] for i in range(cls.num_cls)})

        sum8 = (gt_cls[:, -1:] * (gt["ndsm"]>0)).sum()
        ae8 = (torch.abs(gt["ndsm"] - pred["ndsm"]) * gt_cls[:, -1:] * (gt["ndsm"]>0)).sum()
        se8 = ((gt["ndsm"] - pred["ndsm"])**2 *gt_cls[:, -1:] * (gt["ndsm"]>0)).sum()
        metric_params.update({
            "sum8>0": sum8,
            "ae8>0": ae8,
            "se8>0": se8
        })

        if cls.test:
            assert "mask" in gt, "During test, Ground Truth should contain masks."
            assert gt["mask"].shape[0] == 1, "During test, batch size should be 1."
            pred_l1, gt_l1, pred_h, gt_h = compute_median(gt["mask"].squeeze(), pred["ndsm"].squeeze(), gt["ndsm"].squeeze())
            pred_h0 = pred_h[:, None].repeat(1,  num_bin)
            gt_h0 = gt_h[:, None].repeat(1, num_bin)
            lb = (torch.arange(num_bin)*interval).repeat(len(pred_h), 1).to(pred_h0.device)
            ub = (torch.arange(1, num_bin+1)*interval).repeat(len(gt_h), 1).to(pred_h0.device)
            se_raw = (pred_h0-gt_h0)**2
            discrete_mask = torch.logical_and(gt_h0>=lb, gt_h0<ub)
            count = discrete_mask.sum(dim=0)
            se = (se_raw * discrete_mask).sum(dim=0)
            metric_params.update({
                "sum_mask": gt["mask"].sum(),
                "sum_mask_inv": (1-gt["mask"]).sum(),
                "ae_mask": (torch.abs(pred["ndsm"]-gt["ndsm"])*gt["mask"]).sum(),
                "ae_mask_inv": (torch.abs(pred["ndsm"]-gt["ndsm"]) * (1-gt["mask"])).sum(),
                "se_mask": (((pred["ndsm"]-gt["ndsm"])*gt["mask"])**2).sum(),
                "se_mask_inv": (((pred["ndsm"]-gt["ndsm"])*(1-gt["mask"]))**2).sum(),
                "ae_building": (torch.abs(pred_l1-gt_l1)*gt["mask"].squeeze()).sum(),
                "se_building": (((pred_l1-gt_l1)*gt["mask"].squeeze())**2).sum(),
                "sum_per_building": pred_h.numel(),
                "ae_per_building": torch.abs(pred_h-gt_h).sum(),
                "se_per_building": ((pred_h-gt_h)**2).sum(),
                "per_building_bin_count": count,
                "per_building_bin_se": se
            })
        return metric_params
    
    def get_metric_params(self, pred, gt):
        return self._get_metric_params(self, pred, gt)
    
    @staticmethod
    def _evaluate(cls, eval_dict):
        eval_res = {
            "mae": eval_dict["ae"] / eval_dict["sum"],
            "rmse": torch.sqrt(eval_dict["se"] / eval_dict["sum"]),
        }
        eval_res.update({"mae"+str(i): eval_dict["ae"+str(i)] / eval_dict["sum"+str(i)] for i in range(cls.num_cls)})
        eval_res.update({"rmse"+str(i): torch.sqrt(eval_dict["se"+str(i)] / eval_dict["sum"+str(i)]) for i in range(cls.num_cls)})
        eval_res.update({
            "mae8>0": eval_dict["ae8>0"] / eval_dict["sum8>0"],
            "rmse8>0": torch.sqrt(eval_dict["se8>0"] / eval_dict["sum8>0"])
        })
        if cls.test:
            eval_res.update({
                "mae_mask": eval_dict["ae_mask"] / eval_dict["sum_mask"],
                "mae_non_mask": eval_dict["ae_mask_inv"] / eval_dict["sum_mask_inv"],
                "rmse_mask": torch.sqrt(eval_dict["se_mask"] / eval_dict["sum_mask"]),
                "rmse_non_mask": torch.sqrt(eval_dict["se_mask_inv"] / eval_dict["sum_mask_inv"]),
                "mae_building": eval_dict["ae_building"] / eval_dict["sum_mask"],
                "rmse_building": torch.sqrt(eval_dict["se_building"] / eval_dict["sum_mask"]),
                "mae_per_building": eval_dict["ae_per_building"] / eval_dict["sum_per_building"],
                "rmse_per_building": torch.sqrt(eval_dict["se_per_building"] / eval_dict["sum_per_building"]),
                "per_building_bin_count": eval_dict["per_building_bin_count"],
                "per_building_bin_rmse": torch.sqrt(eval_dict["per_building_bin_se"] / eval_dict["per_building_bin_count"])
            })
        return eval_res

    def evaluate(self, eval_dict):
        return self._evaluate(self, eval_dict)

    @staticmethod
    def _vis(cls, image, pred, gt, validation=False, image_idx=None, save=None):
        prefix = 'val/' if validation else 'train/'
        if save:
            vis_dir = os.path.join(save, 'ndsm')
            os.makedirs(vis_dir, exist_ok=True)
            filename = os.path.join(vis_dir, f"{image_idx[0]}_ndsm_pred.tif")
            io.imsave(filename, pred["ndsm"][0].cpu().numpy())
            #_, _, pred_h, gt_h = compute_median(gt["mask"].squeeze(), pred["ndsm"].squeeze(), gt["ndsm"].squeeze())
            #torch.save([pred_h, gt_h], image_idx[0]+'_h.pkl')
            return {}
        else:
            return {
                prefix+'image': wandb.Image(image[0].cpu()),
                prefix+'ndsm': {
                    'true': wandb.Image(gt['ndsm'][0].float().cpu().numpy()),
                    'pred': wandb.Image(pred['ndsm'][0].detach().float().cpu().numpy())
                }
            }
    
    def vis(self, image, pred, gt, validation=False, image_idx=None, save=None):
        return self._vis(self, image, pred, gt, validation, image_idx, save)