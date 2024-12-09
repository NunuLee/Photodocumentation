import os
import argparse
import torch
from torch import nn
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision.models import *
from torch.utils.tensorboard import SummaryWriter
import torch.distributed
import datetime
import time
from utils.logger import create_logger
from utils.utils import plot_confusion_matrix
from data.build import build_data_loader
from core.train_loop import train_batch_loop
from core.valid_loop import valid_batch_loop
import logging
from model.MTL_swin_b import *
import warnings


# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234 main.py --model-type "swin_b" --epochs 500 --batch-size 64 --learning-rate 1e-4 --num_workers 16
warnings.filterwarnings('ignore')

class_name = ['01_Esophagus','02_Esophagogastric_junction','03_Duodenum_2nd_portion','04_Duodenum_bulb','05_Antrum','06_Angle','07_Body_LB-MB__pic_with_angle','08_Body_MB-HB__pic_without_angle','09_Retro-Fundus','10_Retro-Cardia','11_Retro-Body__LCside', '12_other1', '13_other2', '14_other3', '15_other4', '16_other5', '17_other6', '18_other7']
class_name2 = ['01_E', '02_EJ', '03_D2', '04_DB', '05_A', '06_an', '07_LB', '08_MB', '09_RF', '10_RC', '11_RB', '12_oth1', '13_oth2', '14_oth3', '15_oth4', '16_oth5', '17_oth6', '18_oth7']


def get_args():
    parser = argparse.ArgumentParser(description='Train CNN/ViT on dataset')
    parser.add_argument("--root", default = '/root/APT/multitask_classification/')
    parser.add_argument("--model-type", default = 'swin_b', type = str)
    parser.add_argument("--image-size", default=224, type = int, help = "input image size")
    parser.add_argument("--epochs", type=int, default = 500)
    parser.add_argument("--batch-size", type= int, default=16)
    parser.add_argument('--optimizer', type = str, default = 'adamw', choices=['adam', 'nadam', 'radam', 'adamw','sgd'], dest = 'opt')   # default adamW
    parser.add_argument("--learning-rate", type=float, default = 1e-4, dest = "lr")
    parser.add_argument("--learning-rate-scheduler", type=str,default="true", dest = "lrs" )
    parser.add_argument("--learning-rate-scheduler-minimum", type = float, default=1e-6, dest="lrs_min")
    parser.add_argument("--tensorboard", default = True, action="store_true")
    parser.add_argument("--multi-gpu", type=str, default = 'true', dest='mgpu', choices=['true', 'false'])
    parser.add_argument("--local_rank", type=int, required=True, help="local rank for DistributedDataParallel")
    parser.add_argument("--seed", type = int, default=0, help = 'fixed random seed')
    parser.add_argument("--num_workers", type = int, default = 0, help = "num_workers")

    return parser.parse_args()

def main(args):

    #################### MODEL SETTING ######################
    start_epoch = 0
    num_epochs = args.epochs
    best_f1 = -1 
    best_loss = 100 
    
    early_stop = 0
    
    output_dir_path = '/'
    csv_path = f'{args.root}/data_csv/'
    
    log_dir = os.path.join(output_dir_path, 'log/')
    pth_dir = os.path.join(output_dir_path, 'models/')
    metrics_dir = os.path.join(output_dir_path, 'metrics/')
    os.makedirs(log_dir,exist_ok=True) 
    os.makedirs(pth_dir,exist_ok=True) 
    os.makedirs(metrics_dir, exist_ok=True)

    logger = create_logger(log_dir, dist_rank = 0, name = '')
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.colorbar').setLevel(logging.WARNING)

    # create tensorboard 
    if args.tensorboard:
        writer_t = SummaryWriter(log_dir = log_dir+'/train')
        writer_v = SummaryWriter(log_dir = log_dir+'/val')
      
    ################ Data Loader ########################

    train_loader = build_data_loader(args.image_size, "trainset", args.batch_size, csv_path, args.num_workers, args.local_rank)
    val_loader = build_data_loader(args.image_size, "testset", 1, csv_path, args.num_workers, args.local_rank, shuffle=False)
    
    ################# MODEL CREATE ############################
    model = MTL_Swin_b()
        
    cnt = 0
    for param in model.parameters():
        cnt += 1
        if cnt > (len(list(model.parameters())) - 10):
            param.requires_grad = True
        else:
            param.requires_grad = False

    ################### GPU DISTRIBUTED ###########################
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    if args.mgpu == "true":
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True, broadcast_buffers=False)

    ################## CREATE OPTIMIZER ##########################
    logger.info(f"optimizer = {args.opt} ==================")
    if args.opt == "nadam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=0.0001)
    elif args.opt == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr = args.lr)
    
    logger.info(f"lr_scheduler = {args.lrs} ......")
    if args.lrs == 'true':
        if args.lrs_min > 0:
            tmax = np.ceil((args.epochs / args.batch_size) * 5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= tmax, eta_min=1e-6)
        else:
            tmax = np.ceil((args.epochs / args.batch_size) * 5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= tmax, eta_min=1e-6)
            
        logger.info(f"lr_scheduler = CosineAnnealingLR operating ......")

    ####################### CRITERION ###########################
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    
    ########################## START TRAINING #############################
    if torch.distributed.get_rank() == 0:
        logger.info(f">>>>>>>>>> training starts!")
        print('======================================================================================')
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        if torch.distributed.get_rank() == 0:
            logger.info(f'----------------------------------------  {epoch} / {num_epochs} -------------------------------------')
            early_stop += 1
            if early_stop > 100:
                break
        
        ##############  TRAIN BATCH LOOP ########################
        train_res = train_batch_loop(train_loader, model, optimizer, criterion1, criterion2,epoch, num_epochs, logger)

        writer_t.add_scalar("f1score", train_res[1], epoch)
        writer_t.add_scalar("f1score2", train_res[2], epoch)
        writer_t.add_scalar("Acc", train_res[5].avg, epoch)
        writer_t.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        writer_t.add_scalar("Loss", train_res[0].avg, epoch)
        writer_t.add_scalar("Loss1", train_res[-3].avg, epoch)
        writer_t.add_scalar("Loss2", train_res[-2].avg, epoch)

        ############### DOCTOR VALID BATCH LOOP ########################
        if torch.distributed.get_rank() == 0:
            val_res = valid_batch_loop(val_loader, model, criterion1, criterion2, epoch, logger, 200)
                    
            val_loss = val_res[0].avg
            f1_score = val_res[1]
            cm = val_res[2]
            report = val_res[3]
            val_acc = val_res[4].avg
        
            writer_v.add_scalar("f1score", f1_score, epoch)
            writer_v.add_scalar("Loss", val_loss, epoch)
            writer_v.add_scalar("Acc", val_acc, epoch)
            
            if args.lrs == "true":
                scheduler.step(f1_score)
            
            if f1_score > best_f1:
                best_f1 = f1_score
                early_stop = 0
                logger.info(f'>>>>>>>>>>>>>>>>>>> best f1score {f1_score} epoch : {epoch} save >>>>>>>>>>>>>>>>>>>>>>')
                torch.save(
                    {
                        "epoch":epoch,
                        "model_state_dict":model.state_dict()
                        if args.mgpu=='false'
                        else model.module.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(),
                        "loss":val_loss,
                        "test_measure_mean": f1_score,
                    },
                    f"{pth_dir}_best_cf1_model.pt"
                )
                with open(metrics_dir + f'best_f1_report.txt', 'w') as text_file:
                    print(f'{epoch}_f1score result\n', file=text_file)
                    print(f'{cm[0][0]:3d}\t{cm[0][1]:3d}', file=text_file)
                    print(f'{cm[1][0]:3d}\t{cm[1][1]:3d}\n', file=text_file)
                    print(report, file=text_file)
                    print(f'\n task1 - 18class f1score : {f1_score:.4f}', file=text_file)
                    print(f'\n task1 - 18class loss : {val_loss:.4f}', file=text_file)
                plot_confusion_matrix(cm, f1_score, target_names = class_name2, title = 'best_f1_Confusion_Matrix', path= metrics_dir)
                        
            if val_loss < best_loss:
                best_loss = val_loss
                early_stop = 0
                logger.info(f'>>>>>>>>>>>>>>>>>>> best loss {val_loss} epoch : {epoch} save >>>>>>>>>>>>>>>>>>>>>>')
                torch.save(
                    {
                        "epoch":epoch,
                        "model_state_dict":model.state_dict()
                        if args.mgpu=='false'
                        else model.module.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(),
                        "loss":val_loss,
                        "test_measure_mean": f1_score,
                    },
                    f"{pth_dir}best_loss_model.pt"
                )
                with open(metrics_dir + f'best_loss_report.txt', 'w') as text_file:
                    print(f'{epoch}_f1score result\n', file=text_file)
                    print(f'{cm[0][0]:3d}\t{cm[0][1]:3d}', file=text_file)
                    print(f'{cm[1][0]:3d}\t{cm[1][1]:3d}\n', file=text_file)
                    print(report, file=text_file)
                    print(f'\n task1 - 18class f1score : {f1_score:.4f}', file=text_file)
                    print(f'\n task1 - 18class loss : {val_loss:.4f}', file=text_file)
                plot_confusion_matrix(cm, f1_score, target_names = class_name2, title = 'best_loss_Confusion_Matrix', path= metrics_dir)
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('ALL Training time {}'.format(total_time_str))
    writer_t.flush()
    writer_v.flush()
    writer_t.close()
    writer_v.close()
    print(f'=========================================TRAINING COMPLETED =====================================')

if __name__ == '__main__':

    args = get_args()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    print('local_rank : ' ,args.local_rank)
    
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    torch.manual_seed(2023)
    np.random.seed(2022)
    cudnn.benchmark = True
    main(args)
    torch.cuda.empty_cache()     

