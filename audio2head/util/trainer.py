import random
import importlib
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import glob
import os
import torch
from pytorch_msssim import ssim

from util.distributed import master_only_print as print
from util.distributed import is_master, master_only
from util.init_weight import weights_init

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_trainer(opt, net_G, net_G_ema, opt_G, sch_G, net_D, opt_D, sch_D,\
                    net_left_eye_D,opt_D_left_eye, sch_D_left_eye,\
                    net_right_eye_D, opt_D_right_eye, sch_D_right_eye,\
                    net_mouth_D, opt_D_mouth, sch_D_mouth,train_dataset):
    module, trainer_name = opt.trainer.type.split('::')  # trainers.face_trainer::FaceTrainer

    trainer_lib = importlib.import_module(module)
    trainer_class = getattr(trainer_lib, trainer_name) # 得到FaceTrainer类
    trainer = trainer_class(opt, net_G, net_G_ema, opt_G, sch_G, net_D, opt_D, sch_D,\
                                net_left_eye_D,opt_D_left_eye, sch_D_left_eye,\
                                net_right_eye_D, opt_D_right_eye, sch_D_right_eye,\
                                net_mouth_D, opt_D_mouth, sch_D_mouth,train_dataset)
    return trainer


def _calculate_model_size(model):
    r"""Calculate number of parameters in a PyTorch network.

    Args:
        model (obj): PyTorch network.

    Returns:
        (int): Number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_scheduler(optimizer, config, iterations=-1):
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size,
                                        gamma=config.gamma, last_epoch=iterations)

    return scheduler


def get_optimizer(opt_opt, net):
    return get_optimizer_for_params(opt_opt, net.parameters())


def get_optimizer_for_params(opt_opt, params):
   
    if opt_opt.type == 'adam':
        opt = Adam(params,
                   lr=opt_opt.lr,
                   betas=(opt_opt.adam_beta1, opt_opt.adam_beta2))
    else:
        raise NotImplementedError(
            'Optimizer {} is not yet implemented.'.format(opt_opt.type))
    return opt




def get_3dmm_loss(pred, gt):
    b, t, c = pred.shape
    xpred = pred.contiguous().view(b * t, c)
    xgt = gt.contiguous().view(b * t, c)
    pairwise_distance = F.pairwise_distance(xpred, xgt)
    loss = torch.mean(pairwise_distance)
    spiky_loss = get_spiky_loss(pred, gt)
    return loss, spiky_loss

def get_spiky_loss(pred, gt):
    b, t, c = pred.shape
    pred_spiky = pred[:, 1:, :] - pred[:, :-1, :]
    gt_spiky = gt[:, 1:, :] - gt[:, :-1, :]
    pred_spiky = pred_spiky.contiguous().view(b * (t - 1), c)
    gt_spiky = gt_spiky.contiguous().view(b * (t - 1), c)
    pairwise_distance = F.pairwise_distance(pred_spiky, gt_spiky)
    return torch.mean(pairwise_distance)

def get_head_loss(pred_just_pose, gt_pose):  
    pd_angle, pd_trans, pd_crop = torch.split(pred_just_pose, [3,3,3], dim=-1)
    gt_angle, gt_trans, gt_crop = torch.split(gt_pose, [3,3,3], dim=-1)
    angle_loss, angle_spiky_loss = get_3dmm_loss(pd_angle, gt_angle)
    trans_loss, trans_spiky_loss = get_3dmm_loss(pd_trans, gt_trans)
    crop_loss, crop_spiky_loss  = get_3dmm_loss(pd_crop, gt_crop)
    ssim_loss = 1-ssim(pred_just_pose.unsqueeze(1), gt_pose.unsqueeze(1))
    loss = 0.1*(trans_spiky_loss + crop_loss + crop_spiky_loss)+10*ssim_loss
    # loss = angle_loss + angle_spiky_loss + trans_loss + trans_spiky_loss + crop_loss + crop_spiky_loss
    return loss

def get_exp_loss(pred_exp,gt_exp,):
    exp_loss, exp_spiky_loss = get_3dmm_loss(pred_exp, gt_exp)
    
    loss = exp_loss+0.001*exp_spiky_loss
    return loss



def load_checkpoint(opt, head_model, which_iter=0):
        if which_iter>0:
            model_path = os.path.join(opt.checkpoints_dir, f'Generator_Epoch_{which_iter:03d}.pt') # opt.checkpoints_dir, opt.name 
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            head_model.load_state_dict(checkpoint['net_head'], strict=False)
            current_epoch = which_iter
            print('load [net_head] from {}'.format(model_path))
            return current_epoch
        else:
            current_epoch = 0
            print('No checkpoint found.')
            return current_epoch
        
        
        

def save_checkpoint(opt, epoch, head_generator):
        r"""Save network weights, optimizer parameters, scheduler parameters
        to a checkpoint.
        """
        
        _save_checkpoint(opt, epoch, head_generator)
        
        
@master_only
def _save_checkpoint(opt, epoch, head_generator):
    r"""Save network weights, optimizer parameters, scheduler parameters
    in the checkpoint.

    Args:
        opt (obj): Global configuration.
        opt_G (obj): Optimizer for the generator network.
        sch_G (obj): Scheduler for the generator optimizer.
        current_epoch (int): Current epoch.
        current_iteration (int): Current iteration.
    """
    save_path = os.path.join(opt.checkpoints_dir, f'Generator_Epoch_{epoch + 1:03d}.pt')
    
    torch.save(
        {
            'net_head': head_generator.state_dict()
        },
        save_path,
    )
    print('Save checkpoint to {}'.format(save_path))
    return save_path