import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
from util.distributed import master_only_print as print




def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')



class VoxDataset(Dataset):
    def __init__(self, opt, is_inference, audio_transform, dynam_3dmm_transform, fixed_3dmm_transform):
        path = opt.path
        self.env = lmdb.open(
            os.path.join(path, str(opt.resolution)),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        list_file = "test_list.txt" if is_inference else "train_list.txt"
        list_file = os.path.join(path, list_file)
        with open(list_file, 'r') as f:
            lines = f.readlines()
            videos = [line.replace('\n', '') for line in lines] # 如['RD_Radio34_009', 'WRA_CarlyFiorina_000', 'WRA_PaulRyan0_001'] 
        
        self.resolution = opt.resolution # 256
        self.semantic_radius = opt.semantic_radius # 半径1吧
        self.video_items  = self.get_video_index(videos) # 列表，每个元素是个字典，键值对是video_name,  num_frame，
        
        raw_data = []
        for video_info in self.video_items:
            with self.env.begin(write=False) as txn:
               
                # audio_feats, 未归一化
                audio_feats_key = format_for_lmdb(video_info['video_name'], 'audio_feats')
                audio_feats_numpy = np.frombuffer(txn.get(audio_feats_key), dtype=np.float32).reshape((video_info['num_frame'],-1))
                
                semantics_key = format_for_lmdb(video_info['video_name'], 'coeff_3dmm')
                semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32).reshape((video_info['num_frame'],-1))
                


         
            if video_info['num_frame']>opt.time_size:
                for i in range(video_info['num_frame'] - opt.time_size // 3 + 1):  # -31
                    
                    raw_data.append({
                        'audio': audio_feats_numpy,
                        'speaker': semantics_numpy,
                        'frame_indexs': list(range(i, min(i + opt.time_size, video_info['num_frame']))), # 这样每次都会append一个90长度的list,每个list递增一个数， 最后不够90了就
                    })
                    
            else:
                raw_data.append({
                        'audio': audio_feats_numpy,
                        'speaker': semantics_numpy,
                        'frame_indexs': list(range(video_info['num_frame'])), # 这样每次都会append一个90长度的list,每个list递增一个数， 最后不够90了就
                    })
                
                
                
        
        self.data = raw_data
        self.dynam_3dmm_transform = dynam_3dmm_transform
        self.fixed_3dmm_transform = fixed_3dmm_transform
        self.audio_transform = audio_transform
 
    def __getitem__(self, idx): # 注意这里的idx很大, 是全部的append后的
        frame_indexs = self.data[idx]['frame_indexs']  # 共90个time size的数值
        audio = torch.from_numpy(np.array(self.data[idx]['audio']))[frame_indexs]  # 这里audio一直在变，但是speaker——video不变
        audio = self.audio_transform(audio)
       
        speaker_video = self.data[idx]['speaker']
        speaker_3dmm_fixed,  speaker_3dmm_dynam  = self.load_3dmm(speaker_video, frame_indexs) # 这个是按idx取3dmm参数，这样音频和参数是一一对应的了
        
        # 归一化
        speaker_3dmm_fixed = self.fixed_3dmm_transform(speaker_3dmm_fixed)
        speaker_3dmm_dynam = self.dynam_3dmm_transform(speaker_3dmm_dynam)
        
        init_3dmm_dynam, target_3dmm_dynam = torch.split(speaker_3dmm_dynam, [1, speaker_3dmm_dynam.size(0) - 1]) # 初始的是第一帧， target是后面的
        
        
        speaker_id = speaker_3dmm_fixed[0,:80] # 第一个打头的id
        init_signal = init_3dmm_dynam # 第一帧的3d参数
        target_signal = target_3dmm_dynam # 第二帧到最后的3d参数
      
        
        return audio, init_signal, target_signal, speaker_id
            

            
    def load_3dmm(self, data, frame_indexs):
        coeff_3dmm = data
        id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:,80:144] #expression
        tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:,224:227] #euler angles for pose
        gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:,254:257] #translation
        crop = coeff_3dmm[:,257:260] #crop param
        
        id_gamma_tex = np.concatenate((id_coeff, gamma, tex_coeff), axis=1)
        angle_exp_trans = np.concatenate((angles, ex_coeff, translation), axis=1)
        angle_exp_trans = np.concatenate((angle_exp_trans, crop), axis=1)
        
        id_gamma_tex    = torch.from_numpy(id_gamma_tex[frame_indexs, :]).float()
        angle_exp_trans = torch.from_numpy(angle_exp_trans[frame_indexs, :]).float()
        return id_gamma_tex, angle_exp_trans
        
                
    def __len__(self):
        return len(self.data)  # 这里测试的时候会变成测试视频的帧数
                
                
        
        
        
    def get_video_index(self, videos):
        video_items = []
        for video in videos:
            video_items.append(self.Video_Item(video)) # 每个Video_Item返回一个字典， 字典有三个键值对，分别是video_name, person_id, num_frame
        
        return video_items    
        
        
    def Video_Item(self, video_name):
        video_item = {}
        video_item['video_name'] = video_name

        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], 'length') # 这是length的key
            length = int(txn.get(key).decode('utf-8'))
        video_item['num_frame'] = length
        
        return video_item
    
    
    def transform_semantic(self, semantic, frame_index):
        index = self.obtain_seq_index(frame_index, semantic.shape[0]) # 27长度的list
        
        coeff_3dmm = semantic[index,...]
        
        # id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:,80:144] #expression
        # tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:,224:227] #euler angles for pose
        # gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:,254:257] #translation
        crop = coeff_3dmm[:,257:260] #crop param

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        return torch.Tensor(coeff_3dmm).permute(1,0)
