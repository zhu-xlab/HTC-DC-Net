import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import List

def gaussian_loss(probs, bin_edges, gt, masked=False, reduction=True):
    with torch.no_grad():
        bin_ind = (gt>bin_edges[:, :-1, None, None]) & (gt<=bin_edges[:, 1:, None, None])
        mode_prob = (probs*bin_ind).sum(dim=1, keepdim=True)
        bin_lower_edge = (bin_ind * bin_edges[:, :-1, None, None]).sum(dim=1, keepdim=True)
        bin_upper_edge = (bin_ind * bin_edges[:, 1:, None, None]).sum(dim=1, keepdim=True)
        std = (bin_upper_edge - bin_lower_edge) / 2. / math.sqrt(2) / torch.erfinv(mode_prob)
        cdf_probs = get_gaussian_cdf(std, gt, bin_edges[:, :-1, None, None], bin_edges[:, 1:, None, None], 0)
        cdf_probs = torch.clamp(cdf_probs, 1e-3, 1-1e-3)
        cdf_probs = F.normalize(cdf_probs, p=1, dim=1)
    probs_masks = (probs>0) if masked else torch.ones_like(probs)
    probs = torch.clamp(probs, 1e-3, 1-1e-3)
    probs = F.normalize(probs, p=1, dim=1)
    if reduction:
        loss = (F.kl_div(torch.log(probs), cdf_probs, reduction='none') * probs_masks).sum(dim=1, keepdim=True).mean()
    else:
        loss = (F.kl_div(torch.log(probs), cdf_probs, reduction='none') * probs_masks).sum(dim=1, keepdim=True)
    return loss

def get_gaussian_cdf(std, mean, lower_edge, upper_edge, mode_prob):
    cdf = 0.5 * (torch.erf((upper_edge - mean) / std / math.sqrt(2)) - torch.erf((lower_edge - mean) / std / math.sqrt(2))) - mode_prob
    return cdf

def laplace_loss(probs, bin_edges, gt, masked=False, reduction=True):
    with torch.no_grad():
        bin_ind = (gt>bin_edges[:, :-1, None, None]) & (gt<=bin_edges[:, 1:, None, None])
        mode_prob = (probs*bin_ind).sum(dim=1, keepdim=True)
        bin_lower_edge = (bin_ind * bin_edges[:, :-1, None, None]).sum(dim=1, keepdim=True)
        bin_upper_edge = (bin_ind * bin_edges[:, 1:, None, None]).sum(dim=1, keepdim=True)
        std = -(bin_upper_edge - bin_lower_edge) / 2. / torch.log(1-mode_prob)
        cdf_probs = get_laplace_cdf(std, gt, bin_edges[:, :-1, None, None], bin_edges[:, 1:, None, None], mode_prob, bin_ind)
        cdf_probs = torch.clamp(cdf_probs, 1e-3, 1-1e-3)
        cdf_probs = F.normalize(cdf_probs, p=1, dim=1)
    probs_masks = (probs>0) if masked else torch.ones_like(probs)
    probs = torch.clamp(probs, 1e-3, 1-1e-3)
    probs = F.normalize(probs, p=1, dim=1)
    if reduction:
        loss = (F.kl_div(torch.log(probs), cdf_probs, reduction='none') * probs_masks).sum(dim=1, keepdim=True).mean()
    else:
        loss = (F.kl_div(torch.log(probs), cdf_probs, reduction='none') * probs_masks).sum(dim=1, keepdim=True)
    return loss

def get_laplace_cdf(std, mean, lower_edge, upper_edge, mode_prob, bin_ind):
    cdf1 = 0.5 * (torch.exp((upper_edge-mean)/std) - torch.exp((lower_edge-mean)/std))
    cdf2 = 0.5 * (torch.exp(-(lower_edge-mean)/std) - torch.exp(-(upper_edge-mean)/std))
    cdf3 = mode_prob
    cdf = bin_ind * cdf3 + (lower_edge>mean) * cdf1 + (upper_edge<mean) * cdf2
    return cdf

def uniform_loss(probs, bin_edges, gt, masked=False, reduction=True):
    with torch.no_grad():
        bin_ind = (gt>bin_edges[:, :-1, None, None]) & (gt<=bin_edges[:, 1:, None, None])
        mode_prob = (probs*bin_ind).sum(dim=1, keepdim=True)
        bin_lower_edge = (bin_ind * bin_edges[:, :-1, None, None]).sum(dim=1, keepdim=True)
        bin_upper_edge = (bin_ind * bin_edges[:, 1:, None, None]).sum(dim=1, keepdim=True)
        std = (bin_upper_edge - bin_lower_edge) / mode_prob
        cdf_probs = get_uniform_cdf(std, gt, bin_edges[:, :-1, None, None], bin_edges[:, 1:, None, None])
        cdf_probs = torch.clamp(cdf_probs, 1e-3, 1-1e-3)
        cdf_probs = F.normalize(cdf_probs, p=1, dim=1)
    probs_masks = (probs>0) if masked else torch.ones_like(probs)
    probs = torch.clamp(probs, 1e-3, 1-1e-3)
    probs = F.normalize(probs, p=1, dim=1)
    if reduction:
        loss = (F.kl_div(torch.log(probs), cdf_probs, reduction='none') * probs_masks).sum(dim=1, keepdim=True).mean()
    else:
        loss = (F.kl_div(torch.log(probs), cdf_probs, reduction='none') * probs_masks).sum(dim=1, keepdim=True)
    return loss

def get_uniform_cdf(std, mean, lower_edge, upper_edge):
    lower_bound = mean - std / 2.
    upper_bound = mean + std / 2.
    lower_edge = torch.clamp(lower_edge, min=lower_bound, max=upper_bound)
    upper_edge = torch.clamp(upper_edge, min=lower_bound, max=upper_bound)
    return (upper_edge - lower_edge) / std

def dirac_loss(probs, bin_edges, gt, masked=False, reduction=True):
    with torch.no_grad():
        bin_ind = (gt>bin_edges[:, :-1, None, None]) & (gt<=bin_edges[:, 1:, None, None])
        bin_ind = torch.argmax(bin_ind.to(torch.float), dim=1)
    probs = torch.clamp(probs, 1e-3, 1-1e-3)
    probs = F.normalize(probs, p=1, dim=1)
    if reduction:
        loss = F.nll_loss(torch.log(probs), bin_ind)
    else:
        loss = F.nll_loss(torch.log(probs), bin_ind, reduction='none').unsqueeze(1)
    return loss

def bg_loss(probs, bin_edges, gt, masked=False, reduction=True):
    with torch.no_grad():
        bin_centers = 0.5 * (bin_edges[:, :-1, None, None] + bin_edges[:, 1:, None, None])
        bin_ind = (bin_centers<1).to(torch.float)
        prob0 = torch.clamp(probs*bin_ind, 1e-3, 1-1e-3)
        prob0 = F.normalize(prob0, p=1, dim=1)
    probs = torch.clamp(probs, 1e-3, 1-1e-3)
    probs = F.normalize(probs, p=1, dim=1)
    if reduction:
        loss = F.kl_div(torch.log(probs), prob0)
    else:
        loss = F.kl_div(torch.log(probs), prob0, reduction='none').sum(dim=1, keepdim=True)
    return loss