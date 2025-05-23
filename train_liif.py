""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

import argparse
import io
import os
from collections import deque

import torch
import torch.nn as nn
from tqdm import tqdm
from kornia.filters import Sobel  #


import optuna
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils
from test import eval_psnr


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    model.train()
    l1_loss = nn.L1Loss()
    sobel = Sobel().cuda()
    edge_w = float(config.get('edge_weight', 0.0))
    # print(f"[Train] edge_weight = {edge_w}")

    train_loss = utils.Averager()

    dn = config['data_norm']
    i_sub = torch.FloatTensor(dn['inp']['sub']).view(1, -1, 1, 1).cuda()
    i_div = torch.FloatTensor(dn['inp']['div']).view(1, -1, 1, 1).cuda()
    g_sub = torch.FloatTensor(dn['gt']['sub']).view(1, 1, -1).cuda()
    g_div = torch.FloatTensor(dn['gt']['div']).view(1, 1, -1).cuda()

    for batch in tqdm(train_loader, desc="train", leave=False):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp  = (batch['inp'] - i_sub) / i_div
        pred = model(inp, batch['coord'], batch['cell'])
        gt   = (batch['gt'] - g_sub) / g_div

        # pixel L1
        loss_pix = l1_loss(pred, gt)

        # edge L1
        if edge_w > 1e-8:
            B, Q, _ = pred.shape
            H = W = int(Q ** 0.5)                      # assume square patch
            pred_img = pred.view(B, H, W, 3).permute(0, 3, 1, 2)  # (B,3,H,W)
            gt_img   = gt  .view(B, H, W, 3).permute(0, 3, 1, 2)

            edge_pred = sobel(pred_img)
            edge_gt   = sobel(gt_img)
            loss_edge = l1_loss(edge_pred, edge_gt)

            # edge_term = edge_w * loss_edge
            # print(f"[Train] edge_term = edge_weight * loss_edge = {edge_term.item()}")

            loss = loss_pix + edge_w * loss_edge

        else:
            loss = loss_pix

        train_loss.add(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None

    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()




def evaluate_l1(val_loader, model, loss_fn, config):
    model.eval()
    val_loss = utils.Averager()

    dn = config['data_norm']
    inp_sub = torch.FloatTensor(dn['inp']['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(dn['inp']['div']).view(1, -1, 1, 1).cuda()
    gt_sub  = torch.FloatTensor(dn['gt']['sub']).view(1, 1, -1).cuda()
    gt_div  = torch.FloatTensor(dn['gt']['div']).view(1, 1, -1).cuda()

    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                batch[k] = v.cuda()

            inp  = (batch['inp'] - inp_sub) / inp_div
            pred = model(inp, batch['coord'], batch['cell'])
            gt   = (batch['gt'] - gt_sub) / gt_div
            val_loss.add(loss_fn(pred, gt).item())

    return val_loss.item()



def run_once(config_, save_path, trial=None):
    """
    Single full training run:
    - Train for epoch_max epochs;
    - Compute val_L1 after each epoch;
    - Use the average of the last 5 epochs (tail_avg) as the final score, and report tail_avg to Optuna;
    - If a pruner is used, trial.should_prune() will stop early when tail_avg is poor.
    """
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)

    train_loader, val_loader = make_data_loaders()
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    loss_fn   = nn.L1Loss()
    tail_vals = deque(maxlen=5)

    for epoch in range(epoch_start, config['epoch_max'] + 1):
        _ = train(train_loader, model, optimizer)
        if lr_scheduler:
            lr_scheduler.step()

        val_loss = evaluate_l1(val_loader, model, loss_fn, config)
        tail_vals.append(val_loss)
        tail_avg = sum(tail_vals) / len(tail_vals)

        if trial is not None:
            trial.report(tail_avg, step=epoch)
            if trial.should_prune():
                print(f"[{save_path}] Trial pruned at epoch {epoch}, tail_avg={tail_avg:.4f}")
                raise optuna.TrialPruned()

        print(f"[{save_path}] epoch {epoch}/{config['epoch_max']}  "
              f"val_L1={val_loss:.4f} | tail_avg={tail_avg:.4f}")

    return tail_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with io.open(args.config, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
