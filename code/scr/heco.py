import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder, Mp_encoder1
from .sc_encoder import Sc_encoder
from .contrast import Contrast
import torch

class HeCo(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam):
        super(HeCo, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = Mp_encoder(P, hidden_dim, attn_drop)
        self.sc = Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop)      
        self.contrast = Contrast(hidden_dim, tau, lam)

    def forward(self, feats, pos, mps, nei_index):  # p a s
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        z_mp = self.mp(h_all[0], mps)
        z_sc = self.sc(h_all, nei_index)
        loss = self.contrast(z_mp, z_sc, pos)
        return loss

    def get_embeds(self, feats, mps):
        z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp = self.mp(z_mp, mps)
        return z_mp.detach()

def get_A_r(adj, r):
    adj_label = adj  #delete to_dense() as  'list' object has no attribute 'to_dense'
    if r == 1:
        adj_label = adj_label
    elif r == 2:
        adj_label = adj_label+adj_label
    elif r == 3:
        adj_label = adj_label@adj_label@adj_label
    elif r == 4:
        adj_label = adj_label@adj_label@adj_label@adj_label
    return adj_label

def Ncontrast(x_dis, adj_label, tau = 1):
    """
    compute the Ncontrast loss
    adj_label:(indices=tensor([[   0,    8,   20,  ..., 3992, 4017, 4018],
                       [   0,    0,    0,  ..., 4017, 4017, 4018]]),
       values=tensor([0.0500, 0.0477, 0.0500,  ..., 0.5000, 0.5000, 1.0000]),
       device='cuda:0', size=(4019, 4019), nnz=57853, layout=torch.sparse_coo);
    """
    x_dis = torch.exp(tau * x_dis) #x_dis tensor(4019, 4019)
    x_dis_sum = torch.sum(x_dis, 1) #x_dis_sum tensor 4019 , sum for row of x_dis
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1) #  only integer tensors of a single element can be converted to an index
    #loss = -torch.log(x_dis_sum* (x_dis_sum**(-1))+1e-8).mean()
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

class HeCo1(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam):
        super(HeCo1, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = Mp_encoder1(P, hidden_dim, attn_drop)
        self.sc = Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop)
        self.contrast = Contrast(hidden_dim, tau, lam)

    def forward(self, feats, pos, mps, nei_index, pap1, psp1):  # p a s   #also change add pap1, psp1
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        z_mp, x_dis = self.mp(h_all[0], mps)
        z_sc = self.sc(h_all, nei_index)
        loss = self.contrast(z_mp, z_sc, pos)    #adj_label = get_A_r(mps, 2)
        loss_Ncontrast1 = Ncontrast(x_dis, pap1, tau=1.0)
        loss_Ncontrast2 = Ncontrast(x_dis, psp1, tau=1.0)
        loss = loss+ loss_Ncontrast1 + loss_Ncontrast2

        return loss

    def get_embeds(self, feats, mps):
        z_mp = F.elu(self.fc_list[0](feats[0])) #tensor (4019, 64) cuda grad_fn=<EluBackward0>
        z_mp = self.mp(z_mp, mps) #become tuple
        return z_mp.detach()   #tuple object no attribute detach
    #tuple:(tensor([[ 0.0640, -1.4158,  0.9754,  ..., -2.2123,  2.1256, -1.5471],
""" [ 1.6061,  0.3109, -0.6716,  ..., -1.0086,  0.6214,  1.4109],
        [ 1.5515,  1.5116, -0.3424,  ..., -0.5252,  0.2007,  2.0800],
        ...,
        [-0.3332, -2.1883,  0.1793,  ..., -3.0830,  2.0194, -1.2700],
        [-1.7001, -2.7412, -0.6893,  ..., -1.1199,  0.3304, -3.0207],
        [-1.8518, -0.4678,  0.8673,  ...,  1.2523, -0.5999, -1.0922]],
       device='cuda:0', grad_fn=<AddBackward0>), tensor([[ 0.0000,  0.3335, -0.1938,  ...,  0.9459,  0.5904, -0.9921],
        [ 0.3335,  0.0000,  0.8521,  ...,  0.6151,  0.9466, -0.4086],
        [-0.1938,  0.8521,  0.0000,  ...,  0.1254,  0.6566,  0.1121],
        ...,
        [ 0.9459,  0.6151,  0.1254,  ...,  0.0000,  0.8172, -0.9648],
        [ 0.5904,  0.9466,  0.6566,  ...,  0.8172,  0.0000, -0.6517],
        [-0.9921, -0.4086,  0.1121,  ..., -0.9648, -0.6517,  0.0000]],
       device='cuda:0', grad_fn=<MulBackward0>))"""
