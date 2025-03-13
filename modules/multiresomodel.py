import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fairseq

from .gmlp.gmlp import GMLPBlock
from .p2sgrad.p2sgrad import P2SActivationLayer, P2SGradLoss

#### Mean
class Mean(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.mean(self.dim)

class Max(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return torch.amax(x, self.dim)


####Gmlp
class gMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, seq_len: int, gmlp_layers = 1, batch_first=True, flag_pool='none'):
        """
        gmlp_layer: number of gmlp layers
        d_model: dim(d) of input [n * d]
        d_ffn: dim of hidden feature
        seq_len: the max of input n. for mask 
        batch_first: 

        """
        super().__init__()

        self.batch_first = batch_first

        if(d_ffn > 0):
            pass
        elif(d_ffn < 0):
            #if emb_dim <0, we will reduce dim by emb_dim. like -2 will be dim/2
            d_ffn = int(d_model / abs(d_ffn))

        layers = []
        for i in range(gmlp_layers):
            layers.append(GMLPBlock(d_model, d_ffn, seq_len))
        self.layers = nn.Sequential(*layers)

        if flag_pool == 'mean':
            self.pool = Mean(1)
        elif flag_pool == 'max':
            self.pool = Max(1)
        else:
            self.pool = None

        self.fc = nn.Linear(d_model, d_ffn, bias=False)

    def forward(self, x):

        if(self.batch_first):
            x = x.permute(1, 0, 2)
            x = self.layers(x)
            x = x.permute(1, 0, 2)
        else:
            x = self.layers(x)

        if self.pool is not None:
            x = self.pool(x)

        x = self.fc(x)
        return x    

class SSLModel(nn.Module):
    def __init__(self, ssl_path='xlsr2_300m.pt', ssl_dim=1024, device='cuda'):
        super(SSLModel, self).__init__()

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ssl_path])
        self.model   = model[0]
        self.device  = device
        self.out_dim = ssl_dim

    def extract_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)

        self.model.train()
        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, 0, :]
            else:
                input_tmp = input_data
                
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb

    def extract_layers_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)

        self.model.train()
        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, 0, :]
            else:
                input_tmp = input_data

            results = self.model(input_tmp, mask=False, features_only=True)
            # print(results)
            hidd_states = [ h[2].transpose(0, 1) for h in results["layer_results"] ]
        return hidd_states


class MultiResoModel(torch.nn.Module): # Wav2Vec2 reso is 20ms per frame
    def __init__(self, num_scales=1, num_gmlp_layers=5, include_utt=True, flag_pool='mean', use_mask=False,
                        ssl_dim=1024, ssl_path='xlsr2_300m.pt', ssl_tuning=True, max_seq_len = 2001, device='cuda'):
        super(MultiResoModel, self).__init__()

        self.device = device
        self.num_scales  = num_scales
        self.include_utt = include_utt
        self.featdim     = ssl_dim # Depend on the SSL model
        self.use_mask = use_mask

        self.ssl = SSLModel(device=device, ssl_path=ssl_path, ssl_dim=ssl_dim)
        self.ssl_layer_weights = nn.Parameter(torch.ones(30, device=device))
        self.ssl_tuning = ssl_tuning

        ds_blocks = []
        cl_blocks = []
        ac_blocks = []
        hdim = self.featdim
        for i in range(num_scales):
            hdim = self.featdim // pow(2, i)
            edim = hdim // 2
            seq_len = max_seq_len // pow(2,i)
            # Downsampling
            if i == 0:
                ds_blocks.append(nn.Identity())
            else:
                ds_blocks.append(
                    nn.Sequential(
                        nn.MaxPool2d([2,2], [2, 2]),
                        nn.Linear(hdim, hdim)
                    )
                )
            # Classifying
            cl_blocks.append(
                gMLP(hdim, edim, seq_len, gmlp_layers = num_gmlp_layers,
                    batch_first=True)
            )
            # Activation Function
            ac_blocks.append(P2SActivationLayer(edim, 2))

        self.ds_blocks = nn.ModuleList(ds_blocks)
        self.cl_blocks = nn.ModuleList(cl_blocks)
        self.ac_blocks = nn.ModuleList(ac_blocks)

        self.cl_utt = None
        self.ac_utt = None
        if include_utt:
            self.cl_utt = nn.Sequential(
                nn.Dropout(0.7),
                gMLP(hdim, edim, seq_len, gmlp_layers = num_gmlp_layers,
                    batch_first=True, flag_pool=flag_pool)
            )
            self.ac_utt = P2SActivationLayer(edim, 2)

        self.mask_pool = torch.nn.MaxPool1d(int(16000*0.025), stride=int(16000*0.02))
        self.mask_pool_down = torch.nn.MaxPool1d(2,2)


    def extract_ssl_feat(self, x):
        # print(f"Waveform shape {x.shape}")
        if self.ssl_tuning:
            outs = self.ssl.extract_layers_feat(x.squeeze(1))
        else:
            with torch.no_grad():
                outs = self.ssl.extract_layers_feat(x.squeeze(1))
        nlayers = len(outs)
        norm_weights = nn.functional.softmax(self.ssl_layer_weights[:nlayers], dim=-1)

        feat = torch.stack(outs, dim=-1)
        feat = feat * norm_weights
        feat = torch.sum(feat, dim=-1)
        # print(f"Number of hidden layers {len(outs)} {outs[0].shape} {feat.shape}")

        return feat

    def forward(self, x):
        logits = []
        masks  = []

        #print(f"Waveform {x.shape}")
        # hidd = self.ssl.extract_feat(x)
        hidd = self.extract_ssl_feat(x)

        #print(f"Feat {hidd.shape}")
        for i in range(self.num_scales):
            ds_block, cl_block, ac_block = self.ds_blocks[i], self.cl_blocks[i], self.ac_blocks[i]
            hidd  = ds_block(hidd)
            # print(f"Block {i} - {hidd.shape}")
            logit = ac_block(torch.flatten(cl_block(hidd), start_dim=0, end_dim=1))
            logits.append(logit)
            if self.use_mask:
                if i==0:
                    mask = self.mask_pool(torch.abs(x).squeeze(1))
                else:
                    mask = self.mask_pool_down(mask)
                masks.append(torch.flatten(mask)>0)

        if self.include_utt:
            logit = self.ac_utt(self.cl_utt(hidd))
            logits.append(logit)
            if self.use_mask:
                mask = torch.ones((logit.shape[0],), device=self.device)
                masks.append(mask>0)

        return logits, masks

    def get_num_scales(self):
        if self.cl_utt is None:
            return self.num_scales
        return self.num_scales + 1
