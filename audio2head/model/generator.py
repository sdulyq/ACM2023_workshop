import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.rnn as rnn_utils
import random


def obtain_seq_index(index, num_frames):
        seq = list(range(index-1, index+1+1)) # 27长度的list
        seq = [ min(max(item, 0), num_frames-1) for item in seq ] # 范围限制在0和最大值之间
        # 这个就保证了长度肯定是27， 比如[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]，当index为1
        return seq

class AudioToPose(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.dynam_3dmm_split = [3, 64, 3, 3]
        self.layernorm_init = nn.LayerNorm(param.dynamic_3dmm)
        self.layernorm = nn.LayerNorm(param.audio_dim)
        self.fusion = nn.Sequential(
            nn.Linear(param.audio_dim, param.audio_dim), 
            nn.Tanh(),
        )
        
        self.lstm_layers = nn.LSTM( # lstm_input_size为192，hidden_size 为96， num_layers为4， batch_first为True， dropout为0.1， 
            param.audio_dim,
            param.audio_fusion,
            num_layers=4,
            bias=False,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        
        
        self.lstm = nn.LSTM(param.audio_fusion*3+param.pose_emb*5, param.pose_emb*5, num_layers=2,bias=True,batch_first=True)
        
        
        self.get_initial_state = nn.Sequential(
            nn.Linear(param.dynamic_3dmm, param.dynamic_3dmm),
            nn.Tanh(),
            nn.Linear(param.dynamic_3dmm, param.dynamic_3dmm),
        )

        self.emb_audio = nn.Linear(param.audio_fusion*2, param.audio_fusion)
        self.emb_pos = nn.Linear(param.pose_emb, param.pose_emb*5) # 到45了
        self.output = nn.Linear(param.pose_emb*5, param.pose_emb)
        
        
    def forward(self, audio, init, lengths, teacher_forcing_ration, target=None):
        bs,seqlen,_ = audio.shape
        lengths = lengths.cpu().tolist()
        
        
        
        
        # audio embeding
        audio = self.layernorm(audio)
        audio = self.fusion(audio)
        audio = rnn_utils.pack_padded_sequence(audio, lengths, batch_first=True, enforce_sorted=False) # lengths为90或者小于90的
        audio, _ = self.lstm_layers(audio) # 这里取的是一个序列，不是最后的hidden
        audio, length_unpacked = rnn_utils.pad_packed_sequence(audio, batch_first=True)
        audio_em = self.emb_audio(audio)
        
               
        # pos参数embeding
        ori__angle, ori_exp, orig_trans, ori_crop = torch.split(init, self.dynam_3dmm_split, dim=-1)
        ori_pos = torch.cat(( ori__angle,orig_trans,ori_crop),1).unsqueeze(1)
        
        init = self.get_initial_state(self.layernorm_init(init))
        init_angle, init_exp, init_trans, init_crop = torch.split(init, self.dynam_3dmm_split, dim=-1)
        init_pos = torch.cat((init_angle, init_trans, init_crop),1).unsqueeze(1) # bs, 9, 初始pos
        tem_pos = self.emb_pos(init_pos) # 变成45了
        
        
        # 逐个预测
        zero_state = torch.zeros((2,bs,self.param.pose_emb*5),requires_grad=True).cuda()
        result = [ori_pos]
        # 定义初始值
        cur_state = (zero_state,zero_state)

        for i in range(1,seqlen): # 不预测第一帧
            teacher_force = random.random() < teacher_forcing_ration
            index = obtain_seq_index(i,audio_em.shape[1])
            
           
         
            tem_pos, cur_state = self.lstm(torch.cat((audio_em[:,index].flatten(1).unsqueeze(1),tem_pos),dim=-1), cur_state)
            result.append(self.output(tem_pos))
            tem_pos = self.emb_pos(target[:,i:i+1]) if teacher_force else tem_pos
        res = torch.cat(result, dim=1)
        
        
           
        return res

class PoseGenerator(nn.Module):
    def __init__(self,param):
        super().__init__()
        self.audio_to_pose = AudioToPose(param)
        # self.dynam_3dmm_split = [3, 64, 3, 3]
        
    def predict(self,audio,init,lengths,teacher_forcing_ration, target=None):
        just_pose = self.audio_to_pose(audio, init, lengths, teacher_forcing_ration, target=target)
        return just_pose
    


    def forward(self,audio,init,lengths,teacher_forcing_ration,target=None,):
        just_pose = self.predict(audio, init, lengths,teacher_forcing_ration,target)       
            
        return just_pose
        
        
        


            
            