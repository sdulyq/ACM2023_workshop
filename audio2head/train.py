import data as Dataset
import argparse
from config import Config
from util.trainer import get_scheduler, set_random_seed, get_head_loss, load_checkpoint,save_checkpoint
from util.distributed import init_dist
from model.generator import PoseGenerator
import torch
import torch.nn as nn
import torch.optim as optim
import os
from util.distributed import master_only_print as print
from collections import abc as container_abcs 
from torch._six import  string_classes


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/face.yaml')
    parser.add_argument('--name', default=None)  # 决定用哪个配置文件吗？
    parser.add_argument('--checkpoints_dir', default='result',help='Dir for saving logs and models.')
    parser.add_argument('--which_iter', type=int, default=0)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
   # time_size 在yaml的data下面
    

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    '''
    python -m torch.distributed.launch --nproc_per_node=3 --master_port 12425 train.py
    '''

    # get training options
    dynam_3dmm_split = [3, 64, 3, 3]
    args = parse_args()
    opt = Config(args.config, args, is_train=True) # 里面有很多设置

    if not args.single_gpu:
        opt.local_rank = args.local_rank # 取决于 --nproc_per_node参数， 设置为3则为0，2，1
        init_dist(opt.local_rank)    
        opt.device = opt.local_rank
    else:
        opt.device ='cuda'

    # dataloader
    val_dataset, train_dataset = Dataset.get_train_val_dataloader(opt.data)
    # init model
    head_generator = PoseGenerator(opt.head_model).to(opt.device)
    head_generator_optimizer = optim.AdamW(head_generator.parameters(), lr=opt.head_optimizer.lr,
                            betas=(opt.head_optimizer.adam_beta1,opt.head_optimizer.adam_beta2), weight_decay=opt.head_optimizer.weight_decay)
    
    lr_scheduler_head = get_scheduler(head_generator_optimizer, opt.head_optimizer)
    if opt.distributed: 
        head_generator = nn.SyncBatchNorm.convert_sync_batchnorm(head_generator)
        head_generator = nn.parallel.DistributedDataParallel(
            head_generator,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True, 
            )

    # teacher_forcing
    current_epoch = 0 
    current_epoch = load_checkpoint(opt,head_generator,args.which_iter)
    teacher_forcing_ration = 1


    for epoch in range(current_epoch, opt.max_epoch):
        print('Epoch {} ...'.format(epoch))

        if teacher_forcing_ration <= 0:
            teacher_forcing_ration = 0
        else:
            teacher_forcing_ration -= 0.05

        if not args.single_gpu:
            train_dataset.sampler.set_epoch(current_epoch)

        
        for it, data in enumerate(train_dataset):
            audio, init_signal, target_signal, speaker_id, lengths = data 
            audio = audio.cuda().float()
            init_signal = init_signal.cuda().float()
            target_signal = target_signal.cuda().float()
            lengths = lengths.cuda().long()
            
            
            # gt
            dis_target_signal = torch.cat((init_signal.unsqueeze(1), target_signal),1)
            dis_angle, dis_exp, dis_trans, dis_crop = torch.split(dis_target_signal, dynam_3dmm_split, dim=-1)
            dis_pos = torch.cat((dis_angle, dis_trans, dis_crop),-1)
            
            # for head
            head_generator_optimizer.zero_grad()
            # 这里·····························································
            pred_pose = head_generator(audio, init_signal, lengths,teacher_forcing_ration,target=dis_pos)
            head_loss = get_head_loss(pred_pose[:,1:,:], dis_pos[:,1:,:])
            head_loss.backward()
            head_generator_optimizer.step()
            
            
            if it % 20 == 0:
                print('epoch: ', epoch, '\n')
                print('head_loss: ', head_loss.item())
                print('teacher_forcing_ration: ',teacher_forcing_ration)
                print('learning_rate: ', opt.head_optimizer.lr)
                print(it, '  /  ', len(train_dataset))
                
                  
        save_checkpoint(opt, epoch, head_generator) # 每个epoch保存一次
        
        current_epoch += 1
        lr_scheduler_head.step()