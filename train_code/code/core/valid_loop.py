import torch
import torch.nn.functional as F
import torch
from timm.utils import AverageMeter
import time
import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def accuracy(y_pred, y_true):
    
    y_pred = F.softmax(y_pred, dim = 1)
    top_p, top_class = y_pred.topk(1, dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return (torch.mean(equals.type(torch.FloatTensor))).item()


def valid_batch_loop(data_loader, model, criterion1, criterion2, epoch, logger, print_freq=100):
    model.eval()
    batch_time = AverageMeter()
    valid_loss = AverageMeter()
    valid_acc = AverageMeter()

    gt_labels = []
    pred_labels = []
    
    start = time.time()
    end = time.time()
    for idx, batch in enumerate(data_loader):

        images = batch["image"].cuda(non_blocking=True)
        labels = batch["label"].cuda(non_blocking=True)
        labels2 = batch["label2"].cuda(non_blocking = True)

        logits1, logits2 = model(images)
        loss1 = criterion1(logits1, labels)
        loss2 = criterion2(logits2, labels2)
        
        loss = (0.7 * loss1) + (0.3 * loss2)

        valid_loss.update(loss.item())
        
        valid_acc.update(accuracy(logits1, labels))

        logits1 = F.softmax(logits1, dim = 1)
        _, top_class = logits1.topk(1, dim = 1)
        
        if labels.item() > 10:
            labels = 11 
        else:
            labels = labels.item()
            
        if top_class.item() > 10:
            top_class = 11
        else:
            top_class = top_class.item()
            
        gt_labels.append(labels)
        pred_labels.append(top_class)
           
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Evaluate: [{idx:04d}/{len(data_loader):04d}]\t'
                f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {valid_loss.val:.4f} ({valid_loss.avg:.4f})\t'  
                f'Mem {memory_used:.0f}MB')

    cm = confusion_matrix(gt_labels, pred_labels)
    report_save = classification_report(gt_labels, pred_labels,digits=4, zero_division=True)
    report = classification_report(gt_labels, pred_labels,digits=4, output_dict=True, zero_division=True)
    f1_score = report['weighted avg']['f1-score']
    
    logger.info(f"{epoch} Epoch's f1 score : {f1_score:.4f}")
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} validation takes {datetime.timedelta(seconds=int(epoch_time))}")

    return [valid_loss, f1_score, cm, report_save, valid_acc]
