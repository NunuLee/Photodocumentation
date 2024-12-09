import os
import torch
from matplotlib import pyplot as plt
import itertools
import numpy as np 

def load_checkpoint_files(ckpt_path, model, lr_scheduler, logger):
    print(f">>>>>>>>>>>>> Resuming training from checkpoint: {ckpt_path}")
    logger.info(f">>>>>>>>>>>>> Resuming training from checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt["epoch"] + 1
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    best_loss = ckpt["best_loss"]
    best_acc = ckpt["best_acc"]
    best_f1score = ckpt["best_f1score"]

    del ckpt
    torch.cuda.empty_cache()

    return start_epoch, best_loss, best_acc, best_f1score

def load_checkpoint_files_test(ckpt_path, model):
    print(f">>>>>>>>>>>>> model load >>>>>>>>>>>>>>")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    del ckpt
    torch.cuda.empty_cache()
    
    return

def plot_confusion_matrix(confusion_matrix, f1_score, target_names = None, title = 'Confusion_Matrix', path = None):
    
    acc = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))
    confusion_matrix_n = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize = (20,20))
    plt.imshow(confusion_matrix_n, interpolation='nearest', cmap = plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    else:
        tick_marks = np.arange(confusion_matrix_n.shape[0])
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
    
    for i, j in itertools.product(range(confusion_matrix_n.shape[0]), range(confusion_matrix_n.shape[1])):
        plt.text(j, i, f"{confusion_matrix_n[i, j]:0.4f}\n{confusion_matrix[i][j]}\n",
                horizontalalignment="center", verticalalignment = "center", size = 18,
                color="white" if confusion_matrix_n[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\naccuracy = {acc:0.4f}\nf1-score = {f1_score:0.4f}')
    plt.show()
    plt.savefig(os.path.join(path + f'{title}.png'))
    plt.close()

    return None