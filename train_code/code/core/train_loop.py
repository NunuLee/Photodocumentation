import time
import torch
import datetime
from timm.utils import AverageMeter
import torch.nn.functional as F
import torch
from sklearn.metrics import classification_report

def accuracy(y_pred, y_true):
    
    y_pred = F.softmax(y_pred, dim = 1)
    top_p, top_class = y_pred.topk(1, dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return (torch.mean(equals.type(torch.FloatTensor))).item()

def train_batch_loop(data_loader, model, optimizer, criterion1, criterion2, epoch, num_epochs, logger, print_freq=200):

    model.train()
    
    train_loss = AverageMeter()
    train_loss1 = AverageMeter()
    train_loss2 = AverageMeter()
    batch_time = AverageMeter()
    lr_meter = AverageMeter()
    train_acc = AverageMeter()

    pred_labels = []
    gt_labels = []
    
    pred_labels2 = []
    gt_labels2 = []
    
    num_step = len(data_loader)
    start = time.time()
    end = time.time()
    for idx, batch in enumerate(data_loader):
        images = batch["image"].cuda(non_blocking=True)
        labels1 = batch["label"].cuda(non_blocking=True)
        labels2 = batch["label2"].cuda(non_blocking = True)
        
        logits1, logits2 = model(images)
        loss1 = criterion1(logits1, labels1)
        loss2 = criterion2(logits2, labels2)
        
        loss = (0.7 * loss1) + (0.3 * loss2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        train_loss.update(loss.item())
        train_loss1.update(loss1.item())
        train_loss2.update(loss2.item())

        train_acc.update(accuracy(logits1, labels1))
        lr_meter.update(optimizer.param_groups[0]["lr"])
        
        logits1 = F.softmax(logits1, dim = 1)
        _, top_class = logits1.topk(1, dim=1)      
        gt_labels.extend(labels1.tolist())
        pred_labels.extend(top_class.tolist())

        logits2 = F.softmax(logits2, dim = 1)
        _, top_class2 = logits2.topk(1, dim=1)      
        gt_labels2.extend(labels2.tolist())
        pred_labels2.extend(top_class2.tolist())
       
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_step - idx)
            logger.info(
                f'Train: [{epoch:04d}/{num_epochs:04d}][{idx:04d}/{num_step:04d}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {train_loss.val:.4f} ({train_loss.avg:.4f})\t'  
                f'grad_norm(lr) {lr_meter.val:.4f} ({lr_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    
    report = classification_report(gt_labels, pred_labels, digits = 4, output_dict=True, zero_division=True)
    f1_score = report['weighted avg']['f1-score']

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    
    return [train_loss, f1_score, lr_meter, train_acc, train_loss1, train_loss2]
        