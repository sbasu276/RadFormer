# encoding: utf-8
"""
Training implementation
"""
import os
import cv2
import argparse
import numpy as np
import torch
torch.multiprocessing.set_start_method('spawn')
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import GbDataSet, GbUsgDataSet
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from models import RadFormer
from PIL import Image
import pickle

import warnings
warnings.filterwarnings("ignore")

#np.set_printoptions(threshold = np.nan)


N_CLASSES = 3
CLASS_NAMES = ['nrml', 'benign', 'malg']


def parse():
    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument('--img_dir', dest="img_dir", default="data/gb_imgs")
    parser.add_argument('--val_list', dest="val_list", default="data/cls_split/val.txt")
    parser.add_argument('--out_channels', dest="out_channels", default=2048, type=int)
    parser.add_argument('--model_file', dest="model_file", default="agcnn.pkl")
    parser.add_argument('--global_net', dest="global_net", default="resnet50")
    parser.add_argument('--local_net', dest="local_net", default="bagnet33")
    parser.add_argument('--global_weight', dest="global_weight", default=0.55, type=float)
    parser.add_argument('--local_weight', dest="local_weight", default=0.1, type=float)
    parser.add_argument('--fusion_weight', dest="fusion_weight", default=0.35, type=float)
    parser.add_argument('--num_layers', dest="num_layers", default=4, type=int)
    parser.add_argument('--score_file', dest="score_file", default="out_scores")
    args = parser.parse_args()
    return args


def main(args):
    #print('********************load data********************')
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    val_dataset = GbUsgDataSet(data_dir=args.img_dir, 
                            image_list_file=args.val_list,
                            transform=transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    val_loader = DataLoader(dataset=val_dataset, batch_size=1, 
                                shuffle=False, num_workers=0)

    #print('********************load data succeed!********************')

    #print('********************load model********************')
    # initialize model
    
    model = RadFormer(local_net=args.local_net, \
                        global_weight=args.global_weight, \
                        local_weight=args.local_weight, \
                        fusion_weight=args.fusion_weight, \
                        use_rgb=True, num_layers=args.num_layers, pretrain=False).cuda()

    model.load_state_dict(torch.load(args.model_file)) 
    model.float().cuda()
   
    #print('********************load model succeed!********************')

    #print('*******validation*********')
    y_true, pred_g, pred_l, pred_f = validate(model, val_loader, args)
        
    #print_stats(y_true, pred_g, label="Global")
    #print_stats(y_true, pred_l, label="Local")
    print_stats(y_true, pred_f)


def print_stats(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        cfm = confusion_matrix(y_true, y_pred)
        sens = cfm[2][2]/np.sum(cfm[2])
        spec = ((cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1])))
        acc_binary = (cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1] + cfm[2][2])/(np.sum(cfm))
        print("Acc-3class: %.4f Acc-Binary: %.4f Specificity: %.4f Sensitivity: %.4f"%(acc, acc_binary, spec, sens))

def get_pred_label(pred_tensor):
    _, pred = torch.max(pred_tensor, dim=1)
    return pred.item()


def validate(model, val_loader, args):
    model.eval()
    y_true, pred_g, pred_l, pred_f = [], [], [], []
    score = []
    for i, (inp, target, fname) in enumerate(val_loader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inp.cuda())
            target_var = torch.autograd.Variable(target.cuda())

            g_out, l_out, f_out, _ = model(input_var)

            y_true.append(target.tolist()[0])
            pred_g.append(get_pred_label(g_out)) 
            pred_l.append(get_pred_label(l_out)) 
            pred_f.append(get_pred_label(f_out)) 
            
            score.append([y_true[-1], f_out.tolist()[0]])
    
    with open(args.score_file, "wb") as f:
        pickle.dump(score, f)

    return y_true, pred_g, pred_l, pred_f


if __name__ == "__main__":
    args = parse()
    main(args)
