
from __future__ import print_function, division
import cv2
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from skimage.measure import label
from skimage import filters
from .resnet import Resnet50
from .convnet import FmConvNet
from .bagnet import BagNet33, BagNet17, BagNet9


NETWORK_MAPPER = {
    "resnet50": Resnet50,
    "fmconv": FmConvNet,
    "bagnet33": BagNet33,
    "bagnet17": BagNet17,
    "bagnet9": BagNet9,
}


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, p, d_input=None):
        super().__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input
        # Make sure that the embedding dimension of model is a multiple of number of heads
        assert d_model % self.num_heads == 0
        self.d_k = d_model // self.num_heads
        # These are still of dimension d_model. They will be split into number of heads 
        self.W_q = nn.Linear(d_xq, d_model, bias=False)
        self.W_k = nn.Linear(d_xk, d_model, bias=False)
        self.W_v = nn.Linear(d_xv, d_model, bias=False)
        # Outputs of all sub-layers need to be of dimension d_model
        self.W_h = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V):
        batch_size = Q.size(0) 
        k_length = K.size(-2) 
        # Scaling by d_k so that the soft(arg)max doesnt saturate
        Q = Q / np.sqrt(self.d_k)                         # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(Q, K.transpose(2,3))          # (bs, n_heads, q_length, k_length)
        #A = nn_Softargmax(dim=-1)(scores)   # (bs, n_heads, q_length, k_length)
        A = nn.Softmax(dim=-1)(scores)   # (bs, n_heads, q_length, k_length)
        # Get the weighted average of the values
        H = torch.matmul(A, V)     # (bs, n_heads, q_length, dim_per_head)
        return H, A 

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (heads X depth)
        Return after transpose to put in shape (batch_size X num_heads X seq_length X d_k)
        """
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        """
        Combine the heads again to get (batch_size X seq_length X (num_heads times d_k))
        """
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
    
    def forward(self, X_q, X_k, X_v):
        batch_size, seq_length, dim = X_q.size()
        # After transforming, split into num_heads 
        Q = self.split_heads(self.W_q(X_q), batch_size)  # (bs, n_heads, q_length, dim_per_head)
        K = self.split_heads(self.W_k(X_k), batch_size)  # (bs, n_heads, k_length, dim_per_head)
        V = self.split_heads(self.W_v(X_v), batch_size)  # (bs, n_heads, v_length, dim_per_head)
        # Calculate the attention weights for each of the heads
        H_cat, A = self.scaled_dot_product_attention(Q, K, V)
        # Put all the heads back together by concat
        H_cat = self.group_heads(H_cat, batch_size)    # (bs, q_length, dim)
        # Final linear layer  
        H = self.W_h(H_cat)          # (bs, q_length, dim)
        
        return H, A

    
class CNN(nn.Module):
    def __init__(self, d_model, hidden_dim, p):
        super().__init__()
        
        self.k1convL1 = nn.Linear(d_model,    hidden_dim)
        self.k1convL2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        
        x = self.k1convL1(x)
        x = self.activation(x)
        x = self.k1convL2(x)
        return x

    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, conv_hidden_dim, p=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, p)
        self.cnn = CNN(d_model, conv_hidden_dim, p)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
    
    def forward(self, x):
        # Multi-head attention 
        attn_output, attn = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
        # Layer norm after adding the residual connection 
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        # Feed forward 
        cnn_output = self.cnn(out1)  # (batch_size, input_seq_len, d_model)
        #Second layer norm after adding residual connection 
        out2 = self.layernorm2(out1 + cnn_output)  # (batch_size, input_seq_len, d_model)
        return out2, attn #, out1
    
    
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ff_hidden_dim, p=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.enc_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.enc_layers.append(EncoderLayer(d_model, num_heads, ff_hidden_dim, p))
        
    def forward(self, x):
        attns = [] 
        for i in range(self.num_layers):
            x, attn = self.enc_layers[i](x)
            attns.append(attn)
        return x, attns  # (batch_size, input_seq_len, d_model)


class TransformerNet(nn.Module):
    def __init__(self, num_layers=2, out_features=3, d_model=2048, num_heads=16, conv_hidden_dim=128):
        super().__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, conv_hidden_dim)
        self.dense = nn.Linear(d_model, out_features)
        self.softmax = nn.Softmax(dim=1)#nn.Sigmoid()

    def forward(self, global_pool, local_pool):
        global_pool = global_pool.view(global_pool.size()[0], -1)
        local_pool = local_pool.view(local_pool.size()[0], -1)
        x = torch.stack((global_pool, local_pool), dim=1)
        x, attns = self.encoder(x)
        x, _ = torch.max(x, dim=1)
        x = self.dense(x)
        x = self.softmax(x)
        return x, attns


class RadFormer(nn.Module):
    def __init__(self, global_net="resnet50", local_net="bagnet33", num_cls=3, \
                    in_channels=3, out_channels=2048, in_size=(224,224), \
                    global_weight=0.6, local_weight=0.1, fusion_weight=0.3, \
                    num_layers=2, load_local=False, use_rgb=True, pretrain=True):
        super(RadFormer, self).__init__()
        
        self.num_cls = num_cls
        self.size_upsample = in_size
        self.depth = in_channels
        self.global_weight = global_weight
        self.local_weight = local_weight
        self.fusion_weight = fusion_weight
        self.use_rgb = use_rgb
        if use_rgb and in_channels!=3:
            raise ValueError("In channels must be 3 for RGB images")
        if not use_rgb:
            pretrain=False
        if global_net not in ["resnet18", "resnet34"]:
            self.global_branch = NETWORK_MAPPER[global_net](pretrain=pretrain, load_local=load_local, \
                                    num_cls=num_cls, out_channels=out_channels)
        self.local_branch = NETWORK_MAPPER[local_net](pretrain=pretrain, load_local=load_local, \
                                num_cls=num_cls, out_channels=out_channels)
        if global_net == "resnet18":
            model = models.resnet18(pretrained=False, num_classes=3)


        self.num_features = self.global_branch.out_channels
        self.fusion_branch = TransformerNet(num_layers=num_layers, \
                                            d_model=self.num_features, out_features=num_cls)
        self.criterion = nn.CrossEntropyLoss()
        self.normalize = transforms.Normalize(  
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225]
            )
        self.preprocess = transforms.Compose([
               #transforms.Resize((224,224)),
               transforms.Resize((256,256)),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
               self.normalize,
            ])
        self.get_patch = transforms.Compose([
               transforms.Resize((256,256)),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
            ])

        self.cms = None

    def forward(self, x, target=None, vis=False):
        g_out, g_attn, g_pool = self.global_branch(x)
        local_inp, local_im, box = self.attention(x, g_attn)
        #l_out, l_attn, l_pool = self.local_branch(x)
        l_out, l_attn, l_pool = self.local_branch(local_inp)
        f_out, attns = self.fusion_branch(g_pool, l_pool)
        if target != None:
            loss_g = self.criterion(g_out, target)
            loss_l = self.criterion(l_out, target)
            loss_f = self.criterion(f_out, target)
            loss = self.global_weight*loss_g \
                    + self.local_weight*loss_l \
                    + self.fusion_weight*loss_f
            return loss
        else:
            attn_data = {"g_attn": g_attn, "g_pool": g_pool,\
                         "l_attn": l_attn, "l_pool": l_pool,\
                         "attn_reg": local_im, "box": box, "attns": attns}
            return g_out, l_out, f_out, attn_data

    def get_cam_imgs(self, attn_fm):
        # Get cam (depth wise summation)
        cam = attn_fm.reshape(-1, attn_fm.size()[1], attn_fm.size()[2]*attn_fm.size()[3]).sum(axis=1)
        cam = 255*(cam - cam.min(1, keepdim=True)[0])/cam.max(1, keepdim=True)[0]
        ups = torch.nn.Upsample(size=self.size_upsample, mode="bilinear")
        cam_imgs = ups(cam.view(1, attn_fm.size()[0], attn_fm.size()[2], attn_fm.size()[3])).squeeze(0)
        cam_imgs = cam_imgs.to(torch.uint8).detach().cpu().numpy()
        self.cms = cam_imgs
        return cam_imgs

    def bin_image(self, heatmap):
        _, heatmap_bin = cv2.threshold(heatmap , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #_, heatmap_bin = cv2.threshold(heatmap*filters.apply_hysteresis_threshold(heatmap, 50, 120), 0, 255, cv2.THRESH_BINARY)
        #heatmap_bin = cv2.adaptiveThreshold(heatmap, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 121, 2)
        # t in the paper
        #_, heatmap_bin = cv2.threshold(heatmap , 120 , 255 , cv2.THRESH_BINARY)
        return heatmap_bin
    
    def max_connect(self, heatmap):
        labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)    
        max_label = 0
        max_num = 0
        for i in range(1, num+1):
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        lcc = (labeled_img == max_label)
        if max_num == 0:
           lcc = (labeled_img == -1)
        lcc = lcc + 0
        return lcc 

    def attention(self, x, attn_fm):
        cam_imgs = self.get_cam_imgs(attn_fm)
        patches_cuda = torch.FloatTensor().cuda()
        for i, _img in enumerate(cam_imgs): 
            heatmap_bin = self.bin_image(_img)
            heatmap_maxconn = self.max_connect(heatmap_bin)
            heatmap_mask = heatmap_bin*heatmap_maxconn
            ind = np.argwhere(heatmap_mask != 0)
            if len(ind)>0:
                minh = min(ind[:,0])
                minw = min(ind[:,1])
                maxh = max(ind[:,0])
                maxw = max(ind[:,1])
            else:
                minh = 0
                minw = 0
                maxh = self.size_upsample[0]
                maxw = self.size_upsample[1]
            if maxh-minh==0:
                minh = 0
                maxh = self.size_upsample[0]
            if maxw-minw==0:
                minw = 0
                maxw = self.size_upsample[1]
            
            if self.use_rgb:
                inp = x[i].cpu().numpy().reshape(self.size_upsample[0], self.size_upsample[1], self.depth)
                inp = inp[int(224*0.334):int(224*0.667),int(224*0.334):int(224*0.667),:]
                inp = cv2.resize(inp, self.size_upsample)
                inp_crop = inp[minh:maxh, minw:maxw, :]*256
                local_crop = copy.deepcopy(inp_crop)
                #local_im = self.get_patch(Image.fromarray(local_crop.astype('uint8')).convert('RGB'))
                inp_crop = self.preprocess(Image.fromarray(inp_crop.astype('uint8')).convert('RGB'))
                img_var = torch.autograd.Variable(inp_crop.reshape(self.depth, self.size_upsample[0], self.size_upsample[1]).unsqueeze(0).cuda())
                patches_cuda = torch.cat((patches_cuda, img_var), 0)
            else:
                inp = x[i]
                inp_crop = inp[:, minh:maxh, minw:maxw]
                inp_crop = torch.as_tensor(inp_crop)
                ups = torch.nn.Upsample(size=self.size_upsample, mode="bilinear")
                inp_crop = ups(inp_crop.view(1, inp_crop.size()[0], inp_crop.size()[1], inp_crop.size()[2]))
                #img_var = torch.autograd.Variable(inp_crop.reshape(self.depth, self.size_upsample[0], self.size_upsample[1]).cuda())
                img_var = torch.autograd.Variable(inp_crop.cuda())
                patches_cuda = torch.cat((patches_cuda, img_var), 0)
        return patches_cuda, local_crop, (minw, minh, maxw, maxh)

    def load_model(self, weight_file):
        self.net.load_state_dict(torch.load(weight_file))
