import os
import os.path
import argparse
import toml
import datetime
import shutil
import sys
import time

import numpy as np
import torch
import torchaudio

from torch.utils.data import DataLoader

from utils import reproducibility, save_checkpoint, load_checkpoint
from modules.p2sgrad.p2sgrad import P2SGradLoss

from datasets.partialspoof import PartialSpoofDataset, get_truncated_segment
from modules.multiresomodel import MultiResoModel



def get_basename(filepath):
    path, fname = os.path.split(filepath)
    bname, ext  = os.path.splitext(fname)
    return bname


def train_epoch(dataloader, model, optim, device='cuda'):
    num_scales   = model.get_num_scales()
    running_loss = np.zeros(num_scales)
    
    num_total = 0.0
    
    model.to(device)
    model.train()

    #set objective (Loss) functions
    criterion = P2SGradLoss()
    
    nsteps = len(dataloader)
    for step, batch in enumerate(dataloader):
        x = batch[0]
        labels = batch[1:]
       
        batch_size = x.size(0)
        num_total += batch_size
        
        # num_scales = len(labels)
        x = x.to(device)
        logits, masks = model(x)

        losses = []
        sumloss = 0
        for i in range(num_scales):
            logit = logits[i]
            label = torch.flatten(labels[i].to(device))
            if model.use_mask:
                logit = logit[masks[i]]
                label = label[masks[i]]
            loss = criterion(logit, label)

            sumloss = sumloss + loss
            losses.append(loss)

        running_loss += np.array([loss.item() * batch_size for loss in losses])
        print(f"Step {step:05d}/{nsteps:05d} - Loss {sumloss.item():.04f} {labels[-1]}")

        optim.zero_grad()
        sumloss.backward()
        optim.step()
       
    running_loss /= num_total
    
    return running_loss


def evaluate_accuracy(dataloader, model, device='cuda'):
    num_scales   = model.get_num_scales()
    running_loss = np.zeros(num_scales)
    
    num_total = 0.0
    
    model.to(device)
    model.eval()

    #set objective (Loss) functions
    criterion = P2SGradLoss()
    
    nsteps = len(dataloader)

    with torch.inference_mode():
        for step, batch in enumerate(dataloader):
            x = batch[0]
            labels = batch[1:]
           
            batch_size = x.size(0)
            num_total += batch_size
            
            # num_scales = len(labels)
            x = x.to(device)
            logits, masks = model(x)

            losses = []
            for i in range(num_scales):
                logit = logits[i]
                label = torch.flatten(labels[i].to(device))
                if model.use_mask:
                    logit = logit[masks[i]]
                    label = label[masks[i]]
                loss = criterion(logit, label)

                losses.append(loss)

            running_loss += np.array([loss.item() * batch_size for loss in losses])

    running_loss /= num_total
    
    return running_loss

def load_padded_waveforms(filepaths):
    trailing_pad = int(0.005*16000)
    target_size  = 0

    waveforms = []
    Ts        = []
    for filepath in filepaths:
        waveform, sample_rate = torchaudio.load(filepath)
        size = waveform.shape[-1]
        target_size = target_size if target_size > 0 else max(size + trailing_pad, int(0.645*16000))
        waveform = get_truncated_segment(waveform, target_size)
        waveforms.append(torch.unsqueeze(waveform,0))
        Ts.append(size)

    return torch.cat(waveforms), Ts

def produce_evaluation(scp, model, outpath, batch_size=4, units=[0.02], device='cuda'):
    st  = time.time()

    utterances = []
    with open(scp, 'r') as f:
        for line in f:
            path = line.strip()
            size = os.path.getsize(path)
            name = get_basename(path)
            utterances.append({"basename": name, "path": path, "size": size})
    utterances.sort(key=lambda x: x["size"], reverse=True)

    model.to(device)

    startIdx = 0
    model.eval()
    n = model.get_num_scales()-1 if model.include_utt else model.get_num_scales()
    assert n==len(units), f"len(units) should be equal n={n}"
    print(f"Produce score file for {n} scale and include_utt={model.include_utt}")
    with torch.inference_mode():
        while startIdx < len(utterances):
            endIdx = min(startIdx+batch_size,len(utterances))
            B      = endIdx - startIdx
            filepaths = [ utterance["path"] for utterance in utterances[startIdx:endIdx] ]

            waveforms, Ts = load_padded_waveforms(filepaths)
            waveforms = waveforms.to(device)
            logits, masks = model(waveforms)

            for scale in range(n):
                scalepath = f"{outpath}/unit{units[scale]}.score"
                scores = torch.reshape(logits[scale], (B, -1, 2)).cpu().numpy()
                with open(scalepath, 'a+') as f:
                    for idx in range(B):
                        utterance = utterances[startIdx + idx]
                        name      = utterance["basename"]
                        T    = Ts[idx] 
                        nsegs = max(int(T/(0.02*16000) / pow(2,scale)),1)
                        score = scores[idx,:nsegs, :]
                        f.write("\n".join(f"{name} {i} {item[0]:.05f} {item[1]:.05f}" for i, item in enumerate(score))+"\n") 
            if model.include_utt:
                scalepath = f"{outpath}/utt.score"
                scores = logits[-1]
                with open(scalepath, 'a+') as f:
                    for idx in range(B):
                        utterance = utterances[startIdx + idx]
                        name      = utterance["basename"]
                        score = scores[idx, :]
                        f.write(f"{name} {score[0]:.05f} {score[1]:.05f}\n")

            startIdx = endIdx
        duration  = time.time() - st
    print(f"FINISHED: Inference took {duration/60:.02f} min")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MultiResoModel training")
    parser.add_argument("--config", type=str)
    parser.add_argument("--tag", default="", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_trailing_epochs", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--is_eval", action="store_true")
    parser.add_argument("--eval_scp", default=None, type=str)
    parser.add_argument("--eval_tag", default="", type=str)

    args = parser.parse_args()

    reproducibility(args.seed)

    config    = toml.load(args.config)
    modelname = config["name"]


    model_tag  = f"{modelname}{args.tag}"
    model_path = f'exp/{model_tag}'
    if not os.path.exists(model_path):
        os.makedirs(model_path,exist_ok=True)

    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    modules = {}
    modules["model"]     = MultiResoModel( num_scales = len(config["units"]), include_utt=config["include_utt"], use_mask=config["use_mask"],
                                            ssl_path = config["ssl_path"], ssl_dim = config["ssl_dim"], ssl_tuning=config["ssl_tuning"], device=device ).to(device)
    modules["optimizer"] = torch.optim.Adam( modules["model"].parameters(), 
                                             lr=config["optimizer"]["learning_rate"] )
    
    epoch, not_improving, best_loss, best_epoch, epoch_losses = 0, 0, float('inf'), 0, {}
    # Resume training
    if args.resume_ckpt is not None:
        modules, meta = load_checkpoint(args.resume_ckpt, modules)
        epoch         = meta["last_epoch"]+1
        not_improving = meta["not_improving"]
        best_loss     = meta["best_loss"]
        best_epoch    = meta["best_epoch"]
        if "epoch_losses" in meta:
            epoch_losses = meta["epoch_losses"]
        else:
            for i in range(epoch):
                epoch_losses[i] = 100
            epoch_losses[best_epoch] = best_loss
        print(f"Load pretrained model from {args.resume_ckpt} with epoch={epoch}, best_loss={best_loss}, best_epoch={best_epoch}, not_improving={not_improving}")
        print(f"Epoch losses: {epoch_losses}")

    model, optimizer = modules["model"], modules["optimizer"]

    # StepLR
    scheduler = torch.optim.lr_scheduler.StepLR( optimizer=optimizer,
                                                 step_size=config["scheduler"]["step_size"], 
                                                 gamma=config["scheduler"]["lr_decay_factor"] )

    # ====================#
    #   Evaluation START  #
    # ====================#
    if args.is_eval:
        assert args.resume_ckpt is not None, 'ERROR: Eval but does load a checkpoint with resume_ckpt'
        assert args.eval_scp is not None, 'ERROR: Eval but does load eval_scp'

        eval_outpath = f"{model_path}/result_{modelname}_e{epoch-1}{args.eval_tag}"
        if os.path.exists(eval_outpath):
            shutil.rmtree(eval_outpath)
        os.makedirs(eval_outpath, exist_ok=True)
        produce_evaluation(args.eval_scp, model, eval_outpath, batch_size=args.batch_size, units=config["units"], device=device)
        sys.exit(0)
    # ====================#
    #   Evaluation END    #
    # ====================#


    trainset = PartialSpoofDataset(config["dataset"]["scp_train_wav"], config["dataset"]["scp_train_lab"],
                                    random_seek=config["random_seek"], units=config["units"],
                                    segment_duration=config["segment_duration"],
                                    trailing_duration=config["trailing_duration"])
    devset   = PartialSpoofDataset(config["dataset"]["scp_dev_wav"], config["dataset"]["scp_dev_lab"], units=config["units"],
                                    segment_duration=config["segment_duration"],
                                    trailing_duration=config["trailing_duration"])

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    devloader   = DataLoader(devset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    while not_improving < args.max_trailing_epochs:
        print('#====== Start training Epoch-{} ======#'.format(epoch))

        ####     Start Training Epoch     ####
        train_loss = train_epoch(trainloader, model, optimizer, device=device)
        train_loss_sum = np.sum(train_loss)
        scheduler.step()
        print(f"TRAINING: Epoch {epoch} - Total Loss {train_loss_sum:.04f}")
        #### ==== End Training Epoch ==== ####

        val_loss = evaluate_accuracy(devloader, model, device=device)
        val_loss_sum = np.sum(val_loss)
        print(f'VALIDATION: Epoch {epoch} - Total Loss {val_loss_sum:.04f}')
        epoch_losses[epoch] = val_loss_sum

        is_new_best = val_loss_sum<best_loss
        if is_new_best:
            best_loss, best_epoch = val_loss_sum, epoch
            not_improving = 0
        else:
            not_improving +=1
        ckpt_path = f'{model_path}/{epoch}.pth'

        meta = {"last_epoch": epoch, "best_loss": best_loss, "best_epoch": best_epoch, "not_improving": not_improving, "epoch_losses": epoch_losses}
        save_checkpoint(ckpt_path, modules, meta)

        with open(f'{model_path}/validation.log', 'a+') as f:
            if epoch == 0:
                f.write("Epoch\t" + "\t".join(f"{x:.04f}" for x in config["units"]) + "\tUTT\tTotal\tTimestamp\tBest\n")
            log = f"{epoch}\t" + "\t".join(f"{x:.04f}" for x in val_loss.tolist())
            if not config["include_utt"]:
                log += "\t"
            log += f"\t{val_loss_sum:.04f}\t{datetime.datetime.now()}\t"
            log += "x" if is_new_best else ""
            f.write(f"{log}\n")

        for i in range(epoch):
            top5_epoches = [k for k, v in sorted(epoch_losses.items(), key=lambda item: item[1])]
            top5_epoches = top5_epoches[:5]

            if i in top5_epoches:
                continue
            old_ckpt_path = f'{model_path}/{i}.pth'
            if os.path.isfile(old_ckpt_path):
                os.remove(old_ckpt_path)
        epoch+=1
        if epoch>args.max_epochs:
            break

    print(f'Finished training: Total epochs: {epoch}')
