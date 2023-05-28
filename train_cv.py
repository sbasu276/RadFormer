# encoding: utf-8
"""
Training implementation
"""
import os
import cv2
import time
import argparse
import numpy as np
import torch
import json
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
import warnings
warnings.filterwarnings("ignore")
#import neptune.new as neptune

#np.set_printoptions(threshold = np.nan)

N_CLASSES = 3
CLASS_NAMES = ['nrml', 'benign', 'malg']


def parse():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--img_dir', dest="img_dir", default="data/gb_imgs")
    parser.add_argument('--train_list', dest="train_list", default="data/cls_split/train.txt")
    parser.add_argument('--val_list', dest="val_list", default="data/cls_split/val.txt")
    parser.add_argument('--meta_file', dest="meta_file", default="data/res.json")
    parser.add_argument('--out_channels', dest="out_channels", default=2048, type=int)
    parser.add_argument('--epochs', dest="epochs", default=30, type=int)
    parser.add_argument('--save_dir', dest="save_dir", default="expt")
    parser.add_argument('--save_name', dest="save_name", default="attnbag")
    parser.add_argument('--batch_size', dest="batch_size", default=16, type=int)
    parser.add_argument('--lr', dest="lr", default=0.001, type=float)
    parser.add_argument('--load_local', action="store_true")
    parser.add_argument('--pretrain', action="store_true")
    parser.add_argument('--va', action="store_true")
    parser.add_argument('--global_net', dest="global_net", default="resnet50")
    parser.add_argument('--local_net', dest="local_net", default="bagnet33")
    parser.add_argument('--global_weight', dest="global_weight", default=0.55, type=float)
    parser.add_argument('--local_weight', dest="local_weight", default=0.1, type=float)
    parser.add_argument('--fusion_weight', dest="fusion_weight", default=0.35, type=float)
    parser.add_argument('--fusion_type', dest="fusion_type", default="+") #or *
    parser.add_argument('--optim', dest="optim", default="adam")
    parser.add_argument('--num_layers', dest="num_layers", default=2, type=int)
    parser.add_argument('--gpu', dest="gpu", default=0, type=int)
    parser.add_argument('--model_file', dest="model_file", default="model.pkl")
    args = parser.parse_args()
    return args


def get_loader(to_blur, sigma, args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    train_dataset = GbUsgDataSet(data_dir=args.img_dir,
                            image_list_file=args.train_list,
                            to_blur=to_blur,
                            sigma=sigma,
                            transform=transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ]))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=0)

    return train_loader
    

def main(args):
    #print('********************load data********************')
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    with open(args.meta_file, "r") as f:
        df = json.load(f)

    if not args.va:
        train_loader = get_loader(False, 0, args)

    val_dataset = GbUsgDataSet(data_dir=args.img_dir, 
                            image_list_file=args.val_list,
                            #df=df,
                            #train=True,
                            transform=transforms.Compose([
                                #transforms.Resize((224,224)),
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
                        load_local=args.load_local, use_rgb=True, \
                        num_layers=args.num_layers, pretrain=args.pretrain).cuda()

    #model.load_state_dict(torch.load(args.model_file))
    
    #par_model = nn.DataParallel(model, device_ids = [0,1,2,3,4,5])

    #par_model = par_model.to(0)

    model.float().to(args.gpu)
    #cudnn.benchmark = False
   
    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    
    lr_sched = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 1)
    #print('********************load model succeed!********************')

    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    save_model_name = args.save_name
    best_accs = [0, 0, 0]
    best_ep = 0
    best_f1 = 0
    
    """
    # credentials for neptune board
    run = neptune.init(project='sbasu276/attn-bagnet-va-kfold',
            api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNjgzZTk5Yi0xNmFlLTQ4YTAtODBhZS0xOGRmNzdlMTFhMmEifQ==')

    #run.create_experiment(args.expt_name)
    run["parameters"] = {
                        "LR": args.lr,
                        "batch size": args.batch_size,
                        "optimizer": args.optim,
                        "local net": args.local_net,
                        "data": "VA + Full Sized",
                        "num_layers": args.num_layers,
                        "g_weight": args.global_weight,
                        "l_weight": args.local_weight,
                        "f_weight": args.fusion_weight,
                        "f_type": args.fusion_type,
                        "Split ID": args.val_list.split("/")[-1],
                    }

    """
    #print('********************begin training!********************')
    for epoch in range(args.epochs):

        if args.va:
            # Add VA curriculum
            if epoch < 10:
                train_loader = get_loader(True, 16, args)
            elif epoch >=10 and epoch < 15:
                train_loader = get_loader(True, 8, args)
            elif epoch >=15 and epoch < 20:
                train_loader = get_loader(True, 4, args)
            elif epoch >=20 and epoch < 25:
                train_loader = get_loader(True, 2, args)
            elif epoch >=25 and epoch < 30:
                train_loader = get_loader(True, 1, args)
            else:
                train_loader = get_loader(False, 0, args)

        #print('Epoch {}/{}'.format(epoch , args.epochs - 1))
        #print('-' * 10)
        #set the mode of model
        model.train()  #set model to training mode
        running_loss = 0.0
        #Iterate over data
        for i, (global_inp, target, filenames) in enumerate(train_loader):
            global_input_var = torch.autograd.Variable(global_inp.to(args.gpu))
            target_var = torch.autograd.Variable(target.to(args.gpu))
            
            optimizer.zero_grad()
            loss = model(global_input_var, target_var) 
            loss.backward() 
            #loss.mean().backward()
            optimizer.step()  
            #running_loss += loss.data.item()

        #print('Loss: {:.5f}'.format(running_loss/len(train_loader)))

        #print('*******validation*********')
        start_time = time.time()
        y_true, pred_g, pred_l, pred_f = validate(model, val_loader)
        end_time = time.time()
        #y_true, pred_g, pred_l, pred_f = validate(model, val_loader)
        
        #run["Train/Loss"].log(running_loss/len(train_loader))
        
        #acc_g, conf_g = log_stats(y_true, pred_g, label="Global")
        #acc_l, conf_l = log_stats(y_true, pred_l, label="Local")
        #acc_f, conf_f = log_stats(y_true, pred_f, label="Fusion")


        #save
        torch.save(model.state_dict(), save_path+"/"+save_model_name+'_epoch_'+str(epoch)+'.pkl')
        c = confusion_matrix(y_true, pred_f) #conf_f
        acc2, acc3, nml, ben, spec, sens, f1 = get_stats(c)
        #print("Acc2: %s Acc3: %s Nml: %s, Ben: %s, Spec: %s, Sens: %s, f1: %s"%(acc2, acc3, nml, ben, spec, sens, f1))
        #if best_accs[0] < acc_f: #max(acc_g, acc_l, acc_f):
        if best_f1 < f1: #max(acc_g, acc_l, acc_f):
            best_ep = epoch
            best_cfms = [c, confusion_matrix(y_true, pred_g), confusion_matrix(y_true, pred_f)]
            best_f1 = f1
        #if best_accs[0] < acc_f: #max(acc_g, acc_l, acc_f):
        #    best_accs = [acc_f, acc_l, acc_g] #max(acc_g, acc_l, acc_f)
        #    best_ep = epoch
        #    best_cfms = [conf_f, conf_l, conf_g]
        #    torch.save(model.state_dict(), save_path+"/"+save_model_name+'_epoch_'+str(epoch)+'.pkl')
            #torch.save(model.state_dict(), save_path+"/"+save_model_name+'_epoch_'+str(epoch)+'.pkl')
            #print('Best acc model saved!')

        # LR schedular step
        lr_sched.step()  #about lr and gamma
    #print("Best Epoch: ", best_ep)
    #print("Fusion\n", best_cfms[0], best_accs[0])
    #print("Global\n", best_cfms[2], best_accs[2])
    #print("Local\n", best_cfms[1], best_accs[1])
    #print("%.4f"%best_accs[0])
    print("%s, %.4f"%(best_ep, end_time-start_time))
    print("Fusion, ", *get_stats(best_cfms[0]))
    print("Global, ", *get_stats(best_cfms[1]))
    print("Local, ", *get_stats(best_cfms[2]))


def get_stats(c):
    spec = (c[0][0]+c[0][1]+c[1][0]+c[1][1])/(np.sum(c[0])+np.sum(c[1]))
    sens = c[2][2]/np.sum(c[2])
    nml = c[0][0]/np.sum(c[0])
    ben = c[1][1]/np.sum(c[1])
    f1 = (2*spec*sens)/(spec+sens)
    acc3 = (c[0][0]+c[1][1]+c[2][2])/np.sum(c)
    acc2 = ((np.sum(c[0])+np.sum(c[1]))*spec + np.sum(c[2])*sens)/np.sum(c)

    return round(acc2,3), round(acc3,3), round(nml,3), round(ben,3), round(spec,3), round(sens,3), round(f1,3)


def log_stats(y_true, y_pred, label="Eval"):
        acc = accuracy_score(y_true, y_pred)
        cfm = confusion_matrix(y_true, y_pred)
        
        #logobj["%s/Accuracy"%label].log(acc)
        #logobj["%s/Specificity"%label].log((cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1])))
        #logobj["%s/Sensitivity"%label].log(cfm[2][2]/np.sum(cfm[2]))
        
        return acc, cfm


def get_pred_label(pred_tensor):
    _, pred = torch.max(pred_tensor, dim=1)
    return pred.item()


def validate(model, val_loader):
    model.eval()
    y_true, pred_g, pred_l, pred_f = [], [], [], []
    for i, (global_inp, target, filenames) in enumerate(val_loader):
        with torch.no_grad():
            global_input_var = torch.autograd.Variable(global_inp.to(args.gpu))
            target_var = torch.autograd.Variable(target.to(args.gpu))
            
            g_out, l_out, f_out, _ = model(global_input_var)
            
            y_true.append(target.tolist()[0])
            pred_g.append(get_pred_label(g_out))
            pred_l.append(get_pred_label(l_out)) 
            pred_f.append(get_pred_label(f_out)) 

    return y_true, pred_g, pred_l, pred_f


if __name__ == "__main__":
    args = parse()
    main(args)
