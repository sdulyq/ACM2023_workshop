import data as Dataset
import argparse
from config import Config
from util.distributed import init_dist
from model.generator import PoseGenerator
import torch
import torch.nn as nn
import os
from util.distributed import master_only_print as print
import numpy as np
from scipy.io import savemat
import torchvision.transforms as transforms
from util.trainer import load_checkpoint


def replace_coeffs(gt, new_value):
    result = {}
    result['id.gamma.tex'] = gt
    result['angle.exp.trans'] = new_value
    return result


def convert_coeff_to_mat(coeff):
    _id, _gamma, _tex = np.split(coeff['id.gamma.tex'], (80, 27 + 80), axis=1) # (80, 27, 80)
    _angle, _exp, _trans, _crop = np.split(coeff['angle.exp.trans'], (3, 64 + 3, 3 + 64 + 3), axis=1) # (3, 64, 3, 3)
    coeffs = np.concatenate((_id, _exp, _tex, _angle, _gamma, _trans), axis=1)
    transform_params = np.concatenate((np.ones((_crop.shape[0], 2)) * 256., _crop), axis=1)
   
    result = {
        'coeff': coeffs,
        'transform_params': transform_params,
    }
    return result

def prepare_fake_mat(gt_id_gamma_tex, pred):
    fake_speaker = replace_coeffs(gt_id_gamma_tex, pred)
    fake_speaker_mat = convert_coeff_to_mat(fake_speaker)
    return fake_speaker_mat

def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--config', default='./config/face_demo.yaml')
    parser.add_argument('--name', default=None)  # 决定用哪个配置文件吗？
    parser.add_argument('--checkpoints_dir', default='result',help='Dir for saving logs and models.')
    parser.add_argument('--which_iter', type=int, default=150)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output_path', type=str)
    
    
    # time_size 在yaml的data下面
    

    args = parser.parse_args()
    return args




if __name__ == '__main__':
    '''
    python -m torch.distributed.launch --nproc_per_node=1 --master_port 12145 inference.py \
  --input   /root/autodl-tmp/test_with_sadtalker/fir_data/with_audio_vox_lmdb \
  --output_path /root/autodl-tmp/test_with_sadtalker/fir_data/pred_recons
    '''
    
    dynam_3dmm_split = [3, 64, 3, 3]
    args = parse_args()
    opt = Config(args.config, args, is_train=False)
    
    opt.data.path = args.input
    
    if not args.single_gpu:
        opt.local_rank = args.local_rank
        init_dist(opt.local_rank)    
        opt.device = torch.cuda.current_device()
        
    # dataloader
    val_dataset, train_dataset = Dataset.get_train_val_dataloader(opt.data)
    
    head_generator = PoseGenerator(opt.head_model).to(opt.device)
    
    if opt.distributed:
        head_generator = nn.SyncBatchNorm.convert_sync_batchnorm(head_generator)
        head_generator = nn.parallel.DistributedDataParallel(
                head_generator,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True, 
                )
            
    current_epoch = 0 
    current_epoch = load_checkpoint(opt,head_generator,args.which_iter) # load checkpoint
    
    # 恢复原来的一个变换
    dynam_mean_std_info = torch.load(os.path.join(opt.data.path, 'video_feats_mean_std.bin'))['angle.exp.trans']
    dynam_mean, dynam_std = dynam_mean_std_info['mean'], dynam_mean_std_info['std']
    dynam_3dmm_transform = transforms.Lambda(lambda e: e * dynam_std + dynam_mean) # 这是恢复原来的一个变换
    
    fixed_mean_std_info = torch.load(os.path.join(opt.data.path, 'video_feats_mean_std.bin'))['id.gamma.tex']
    fixed_mean, fixed_std = fixed_mean_std_info['mean'], fixed_mean_std_info['std']
    fixed_3dmm_transform = transforms.Lambda(lambda e: e*fixed_std+fixed_mean)
    # save
    mat_output_dir = os.path.join(opt.output_path, 'recon_coeffs', 'test')
    os.makedirs(mat_output_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.output_path, 'vox_lmdb'), exist_ok=True)
    
    
    head_generator.eval()
    test_in_id_count = 1
    teacher_forcing_ration = 0
    
    
    
    with torch.no_grad():   
        if not args.single_gpu:
            val_dataset.sampler.set_epoch(current_epoch)
        for it, data in enumerate(val_dataset):
            
            audio, init_signal, target_signal, speaker_id, lengths,name,id_gamma_tex = data 
            init_signal = init_signal.cuda().float()
            target_signal = target_signal.cuda().float()
            audio = audio.cuda().float()
            lengths = lengths.cuda().long()
            id_gamma_tex = id_gamma_tex.float()
            
            pred_pose = head_generator(audio, init_signal, lengths,teacher_forcing_ration,target=None)
           
            pred_exp = torch.ones((1, pred_pose.shape[1],64)).cuda() # 弄一个假的
            # cat exp&pos
            pd_angle, pd_trans, pd_crop = torch.split(pred_pose,[3,3,3],dim=-1)
            preds = torch.cat((pd_angle,pred_exp,pd_trans,pd_crop), -1)
            
            
            
        
            for name, pred, length, gt_id_gamma_tex in zip(name, preds, lengths, id_gamma_tex):
                # 这里是4个
                
                name = name
                pred = pred[:length].cpu() 
                pred = dynam_3dmm_transform(pred)
                pred_angle, pred_exp, pred_trans, pred_crop = np.split(pred, (3, 64 + 3, 3 + 64 + 3), axis=1) # (3, 64, 3, 3)
                
                gt_id_gamma_tex = fixed_3dmm_transform(gt_id_gamma_tex).cpu()
                # pred = np.concatenate((pred_angle,gt_exp,pred_trans, pred_crop), axis=1)
                
                
                fake_mat = prepare_fake_mat(gt_id_gamma_tex, pred)    
                # fake_mat = prepare_fake_mat(gt_id_gamma_tex, gt_angle_exp_trans) 
                savemat(os.path.join(mat_output_dir, name + '.mat'), fake_mat)
            print(f"TestInId: write image\tepoch -1\tcount {test_in_id_count} / {len(val_dataset)}")
            test_in_id_count += 1
        print('finished!')