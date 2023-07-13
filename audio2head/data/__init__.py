import importlib

import torch.utils.data
from util.distributed import master_only_print as print
import os
import torchvision.transforms as transforms
import numpy as np
import torch.nn.utils.rnn as rnn_utils


def find_dataset_using_name(dataset_name):
    dataset_filename = dataset_name
    module, target = dataset_name.split('::') # data.vox_dataset::VoxDataset
    datasetlib = importlib.import_module(module)
    dataset = None
    for name, cls in datasetlib.__dict__.items():
        if name == target:
            dataset = cls  # 把文件下的VoxDataset类读取了
            
    if dataset is None:
        raise ValueError("In %s.py, there should be a class "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target))
        
    return dataset
        

def create_dataloader(opt, is_inference):
    dataset = find_dataset_using_name(opt.type)  # data.vox_dataset_demo::VoxDataset
    
    # 即读取了data文件夹下的vox_datasetde的VoxDataset类
    
    # audio_transform
    mfcc_mean_std_info = torch.load(os.path.join(opt.path, 'audio_feats_mean_std.bin'))
    mfcc_mean, mfcc_std = mfcc_mean_std_info['mean'], mfcc_mean_std_info['std']
    audio_transform = transforms.Lambda(lambda e: (e - mfcc_mean) / mfcc_std)
    
    # dynamic 3dmm 
    dynam_mean_std_info = torch.load(os.path.join(opt.path, 'video_feats_mean_std.bin'))['angle.exp.trans']
    dynam_mean, dynam_std = dynam_mean_std_info['mean'], dynam_mean_std_info['std']
    dynam_3dmm_transform = transforms.Lambda(lambda e: (e - dynam_mean) / dynam_std)
    
    fixed_mean_std_info = torch.load(os.path.join(opt.path, 'video_feats_mean_std.bin'))['id.gamma.tex']
    fixed_mean, fixed_std = fixed_mean_std_info['mean'], fixed_mean_std_info['std']
    fixed_3dmm_transform = transforms.Lambda(lambda e: (e - fixed_mean) / fixed_std)
    
    
    instance = dataset(opt, is_inference, audio_transform, dynam_3dmm_transform, fixed_3dmm_transform) # 最重要的是里面定义了索引什么的
    
    
    phase = 'val' if is_inference else 'training'
    batch_size = opt.val.batch_size if is_inference else opt.train.batch_size
    print("%s dataset [%s] of size %d was created" %
          (phase, opt.type, len(instance))) # 就是persion_id个数×100后的结果
    
    def collate_fn(batch):
        audio = [e[0] for e in batch]
        init = [e[1] for e in batch]
        target = [e[2] for e in batch]
        speaker_id = [e[3] for e in batch]
        
       
        lengths = torch.from_numpy(np.array([e.size(0) for e in audio]))
        audio = rnn_utils.pad_sequence(audio, batch_first=True)
        target = rnn_utils.pad_sequence(target, batch_first=True) 
        init = torch.vstack(init)
        speaker_id = torch.vstack(speaker_id) 
        
        
      
        return audio, init, target, speaker_id, lengths
    
    def collate_fn_test(batch):
        audio = [e[0] for e in batch]
        init = [e[1] for e in batch]
        target = [e[2] for e in batch]
        speaker_id = [e[3] for e in batch]
        name = [e[4] for e in batch]
        speaker_3dmm_fixed = [e[5] for e in batch]
        
        
        lengths = torch.from_numpy(np.array([e.size(0) for e in audio]))
        

        audio = rnn_utils.pad_sequence(audio, batch_first=True)
        target = rnn_utils.pad_sequence(target, batch_first=True) 
        speaker_3dmm_fixed = rnn_utils.pad_sequence(speaker_3dmm_fixed, batch_first=True) 
        
        init = torch.vstack(init)
        speaker_id = torch.vstack(speaker_id) 
        
      
        return audio, init, target, speaker_id, lengths,name,speaker_3dmm_fixed
    
    dataloader = torch.utils.data.DataLoader(
        instance,
        collate_fn=collate_fn,
        batch_size=batch_size,
        sampler=data_sampler(instance, shuffle=not is_inference, distributed=opt.train.distributed),
        drop_last=not is_inference,
        num_workers=getattr(opt, 'num_workers', 4),
    )        
    
    dataloader_test = torch.utils.data.DataLoader(
        instance,
        collate_fn=collate_fn_test,
        batch_size=batch_size,
        sampler=data_sampler(instance, shuffle=not is_inference, distributed=opt.train.distributed),
        drop_last=not is_inference,
        num_workers=getattr(opt, 'num_workers',0),
    )     

    if not is_inference:
        return dataloader 
    else:
        return dataloader_test
    



def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)



def get_train_val_dataloader(opt):
    train_dataset = create_dataloader(opt, is_inference=False)
    val_dataset = create_dataloader(opt, is_inference=True)
    return val_dataset, train_dataset