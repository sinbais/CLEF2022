import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import torch
import sys
sys.path.append('../pytorch-image-models-master')
import presets
import transforms

import logging
import argparse
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
from PIL import Image
from tqdm import tqdm
import random
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.distributed as dist
import torch.multiprocessing as mp
# from torchvision import transforms
import timm
import time
import torchvision

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, SmallestMaxSize
)

from torch.utils.data.dataloader import default_collate
from albumentations.pytorch import ToTensorV2
import pandas as pd
from torch.nn.parallel import DataParallel
import torch
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply
from sklearn.metrics import f1_score, accuracy_score
from PIL import ImageFile
Image.MAX_IMAGE_PIXELS = int(1024 * 1024 * 1024 // 4)
ImageFile.LOAD_TRUNCATED_IMAGES = True

CFG = {
    'root_dir': '/root/dataset/train/DF20-train_val',
    'fold_num': 5,
    'seed': 68,  # 719,42,68
    'model_arch': 'swinv2',
    # 'model_arch': 'volo_d5',
    # tf_efficientnet_b4_ns-d6313a46, tf_efficientnetv2_l_in21ft1k, resnext50_32x4d;tf_efficientnet_b3_ns
    'img_size': 384,
    'resize_size': 384,
    'crop_size': 384,
    'warmup_epochs': 3,
    'epochs': 15,
    'train_bs': 9,
    'valid_bs': 18,
    'T_0': 15,
    'lr': 1.5e-4,
    'min_lr': 1e-5/7,
    'lr_warmup_decay': 0.01,
    'weight_decay': 2e-5,
    'num_workers': 24,
    'accum_iter': 1,
    'verbose_step': 1,
    'device': 'cuda:0',
    'smoothing': 0.1,
    'cutmix_prob': 0.8,
    'class_num': 1604,
    'ema_decay': 0.99998,
    'ema_steps': 32,
}

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(CFG['model_arch']+'_5flod_mis.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


train_data_root = "/root/autodl-tmp/datasets/fungi/DF20"
train = pd.read_csv('./train_val_all.csv')

cls_num_list = np.zeros(1604, dtype=np.int32)
with open('train_val_all.csv', 'r') as f:
    f.readline()
    for line in f:
        label = int(line.split(',')[0])
        cls_num_list[label] += 1

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg)

class LabelAwareSmoothing(nn.Module):
    def __init__(self, smooth_head=0.3, smooth_tail=0.0, shape='concave', power=None):
        super(LabelAwareSmoothing, self).__init__()

        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)

        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth).unsqueeze(1)
        self.smooth = self.smooth.float()
        if torch.cuda.is_available():
            self.smooth = self.smooth.cuda()

    def forward(self, x, target):
        smoothing = (target @ self.smooth).squeeze(1)
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        #nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        #nll_loss = nll_loss.squeeze(1)
        nll_loss = -(logprobs * target).sum(dim=1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss

        return loss.mean()


class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
       
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
# def get_img(path):
#     im_bgr = cv2.imread(path)
#     im_rgb = im_bgr[:, :, ::-1]
#     return im_rgb

def get_img(path):
    img = Image.open(path).convert('RGB')
    return img

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class FungiDataset(Dataset):
    '''
    fungi dataset
    '''
    def __init__(self, path, root, part='train', transforms=None):
        self.part = part
        self.transforms = transforms
        self.images = []
        self.labels = []
        self.cls_num_list = [0] * CFG['class_num']
        if part=='train':
            data = pd.read_csv(os.path.join(path, 'Train.csv'))
            image_files = list(data['image_path'])
            class_ids = list(data['class_id'])
        elif part=='val':
            data = pd.read_csv(os.path.join(path, 'Val.csv'))
            image_files = list(data['image_path'])
            class_ids = list(data['class_id'])
        root_dir = os.path.join(root, 'DF20')

        for idx, image_file in enumerate(image_files):
            class_id = class_ids[idx]
            img_path = os.path.join(root_dir, image_file)
            self.images.append(img_path)
            self.labels.append(class_id)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        image = get_img(self.images[index])
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        return image, self.labels[index]

class CassavaDataset(Dataset):
    def __init__(self, df, data_root,
                 transforms=None,
                 output_label=True,
                 one_hot_label=False,
                 do_fmix=False,
                 fmix_params={
                     'alpha': 1.,
                     'decay_power': 3.,
                     'shape': (CFG['img_size'], CFG['img_size']),
                     'max_soft': True,
                     'reformulate': False
                 },
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 }
                 ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        
        if output_label == True:
            self.labels = self.df['label'].values
            self.cls_num_list = [0] * CFG['class_num']
            for label in self.labels:
                self.cls_num_list[label] += 1
            # print(self.labels)
            
            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max() + 1)[self.labels]
            # print(self.labels)
    
    def __len__(self):
        return self.df.shape[0]
        #return 1024
    
    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.labels[index]
        img = get_img("{}/{}".format(self.data_root, self.df.loc[index]['image_id']))
        # img = get_img(self.df.loc[index]['image_id'])
        
        if self.transforms:
            # img = self.transforms(image=img)['image']
            img = self.transforms(img)
        
        # print(target)
        # print(type(img), type(target))
        if self.output_label == True:
            return img, target
        else:
            return img

def prepare_dataloader(df, trn_idx, val_idx, data_root=train_data_root):
    train_ = df.loc[trn_idx, :].reset_index(drop=True)
    valid_ = df.loc[val_idx, :].reset_index(drop=True)
    
    # train_ds = CassavaDataset(train_, data_root, transforms=get_train_transforms(), output_label=True,
    #                           one_hot_label=False, do_fmix=False, do_cutmix=False)
    # valid_ds = CassavaDataset(valid_, data_root, transforms=get_valid_transforms(), output_label=True)

    train_ds = CassavaDataset(train_, data_root, transforms=presets.ClassificationPresetTrain(
                crop_size=CFG['crop_size'],
                auto_augment_policy="ta_wide",
                random_erase_prob=0.1,
            ), output_label=True, one_hot_label=False, do_fmix=False, do_cutmix=False)

    valid_ds = CassavaDataset(valid_, data_root, transforms=presets.ClassificationPresetEval(
                crop_size=CFG['crop_size'], resize_size=CFG['resize_size'],
            ), output_label=True)


    mixupcutmix = torchvision.transforms.RandomChoice([
        transforms.RandomMixup(num_classes=CFG['class_num'], p=1.0, alpha=0.2),
        transforms.RandomCutmix(num_classes=CFG['class_num'], p=1.0, alpha=1.0)
    ])
    collate_fn = lambda batch: mixupcutmix(*default_collate(batch))
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=CFG['num_workers'],
        collate_fn=collate_fn,
        # sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader

def get_train_transforms():
    return Compose([
        RandomResizedCrop(CFG['img_size'], CFG['img_size'], interpolation=cv2.INTER_CUBIC, scale=(0.5,1)),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        # ShiftScaleRotate(p=0.3),
        IAAPiecewiseAffine(p=0.5),
        HueSaturationValue(hue_shift_limit=4, sat_shift_limit=4, val_shift_limit=4, p=1.0),
        RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),
        OneOf([
            OpticalDistortion(distort_limit=1.0),
            GridDistortion(num_steps=5, distort_limit=1.),
            # ElasticTransform(alpha=3),
        ], p=0.5),
        # Cutout(max_h_size=int(CFG['img_size'] * 0.05), max_w_size=int(CFG['img_size'] * 0.05), num_holes=5, p=0.5),
        # CoarseDropout(max_holes=8, max_height=int(CFG['img_size'] * 0.05), max_width=int(CFG['img_size'] * 0.05), mask_fill_value=0, p=1.0),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

def get_valid_transforms():
    return Compose([
        Resize(CFG['img_size'], CFG['img_size'], interpolation=cv2.INTER_CUBIC),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

def train_one_epoch(epoch, model, model_ema, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()
    
    t = time.time()
    running_loss = None
    image_preds_all = []
    image_targets_all = []

    '''cls_num_list=torch.Tensor(train_loader.dataset.cls_num_list)
    weight = torch.log(cls_num_list / cls_num_list.sum() + 1e-9)[None, :]
    weight = weight.to(device)'''

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device)#.long()
        with autocast():
            image_preds = model(imgs)  # output = model(input)
            #logits = image_preds + weight
            #logits -= torch.max(logits, 1, True)[0].detach()
            # print(image_preds.shape, exam_pred.shape)
            image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]

            loss = loss_fn(image_preds, image_labels)
            image_labels = torch.topk(image_labels, 1)[1].squeeze(1)
            image_targets_all += [image_labels.detach().cpu().numpy()]
            
            scaler.scale(loss).backward()
            
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01
            
            if model_ema and step % CFG['ema_steps'] == 0:
                model_ema.update_parameters(model)
            
            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler is not None and schd_batch_update:
                    scheduler.step()
            
            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                
                pbar.set_description(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    # print(image_preds_all[:10])
    # print(image_targets_all[:10])
    ans = (image_preds_all == image_targets_all).mean()
    print('Train multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))
    logger.info(' Epoch: ' + str(epoch) + ' Train multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))

    if scheduler is not None and not schd_batch_update:
        scheduler.step()

def train_one_epoch_cm(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()
    running_loss = None
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        
        beta = 1.0
        if np.random.rand(1) < CFG['cutmix_prob']:
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(imgs.size()[0]).cuda()
            target_a = image_labels
            target_b = image_labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
            imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
            with autocast():
                image_preds = model(imgs)
                loss = loss_fn(image_preds, target_a) * lam + loss_fn(image_preds, target_b) * (1. - lam)
                scaler.scale(loss).backward()
        else:
            with autocast():
                image_preds = model(imgs)  # output = model(input)
                loss = loss_fn(image_preds, image_labels)
                scaler.scale(loss).backward()

        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
            
        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * .99 + loss.item() * .01
            
        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
            # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
                
            if scheduler is not None and schd_batch_update:
                scheduler.step()
            
        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
            description = f'epoch {epoch} loss: {running_loss:.4f}'
            pbar.set_description(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    ans = (image_preds_all == image_targets_all).mean()
    print('Train multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))
    logger.info(' Epoch: ' + str(epoch) + ' Train multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))

    if scheduler is not None and not schd_batch_update:
        scheduler.step()

def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()
    
    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        
        image_preds = model(imgs)  # output = model(input)
        # print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
        loss = loss_fn(image_preds, image_labels)
        
        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]
        
        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    ans = (image_preds_all == image_targets_all).mean()
    score = f1_score(image_targets_all, image_preds_all, average='macro')
    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))
    logger.info(' Epoch: ' + str(epoch) + 'validation multi-class accuracy = {:.4f}; Macro F1-score = {:.4f}.'.format((image_preds_all == image_targets_all).mean(), score))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()
    return score

def split_normalization_params(model, norm_classes=None):
    # Adapted from https://github.com/facebookresearch/ClassyVision/blob/659d7f78/classy_vision/generic/util.py#L501
    if not norm_classes:
        norm_classes = [nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm]

    for t in norm_classes:
        if not issubclass(t, nn.Module):
            raise ValueError(f"Class {t} is not a subclass of nn.Module.")

    classes = tuple(norm_classes)

    norm_params = []
    other_params = []
    for module in model.modules():
        if next(module.children(), None):
            other_params.extend(p for p in module.parameters(recurse=False) if p.requires_grad)
        elif isinstance(module, classes):
            norm_params.extend(p for p in module.parameters() if p.requires_grad)
        else:
            other_params.extend(p for p in module.parameters() if p.requires_grad)
    return norm_params, other_params

def load_pretrained(path, model):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if L1 != L2 and nH1 == nH2:
            # bicubic interpolate relative_position_bias_table if not match
            S1 = int(L1 ** 0.5)
            S2 = int(L2 ** 0.5)
            relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                mode='bicubic')
            state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if L1 != L2 and C1 == C1:
            S1 = int(L1 ** 0.5)
            S2 = int(L2 ** 0.5)
            absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
            absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
            absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
            absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
            absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
            state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']

    msg = model.load_state_dict(state_dict, strict=False)
    del checkpoint
    torch.cuda.empty_cache()

from thop import profile
if __name__ == '__main__':
    seed_everything(CFG['seed'])
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(
        np.arange(train.shape[0]), train.label.values)
    for fold, (trn_idx, val_idx) in enumerate(folds):
        # we'll train fold 0 first
        #if fold > 2:
        #    break
        device = torch.device(CFG['device'])
        
        model = timm.create_model(CFG['model_arch'], num_classes=CFG['class_num'], pretrained=True)
        #state_dict = torch.load('/root/autodl-nas/swinv2_large_patch4_window12_192_22k.pth')
        load_pretrained('/root/autodl-nas/swinv2_large_patch4_window12_192_22k.pth', model)
        model = nn.DataParallel(model)
        # model.load_state_dict(torch.load('swin_tiny_patch4_window7_224_cutmix_fold_0.pth'))
        model.to(device)
        
        adjust = CFG['train_bs'] * CFG['ema_steps'] / CFG['epochs']
        alpha = 1.0 - CFG['ema_decay']
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(model.module, device=device, decay=1.0 - alpha)
        #model_ema = None
        # model.load_state_dict(torch.load('{}_epoch_14_fold_{}.pth'.format(CFG['model_arch'], fold)))
        train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx, data_root=train_data_root)

        scaler = GradScaler()
        param_groups = split_normalization_params(model)
        wd_groups = [0.0, CFG['weight_decay']]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

        optimizer = torch.optim.AdamW(parameters, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG['epochs']
        )
        '''warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=CFG['lr_warmup_decay'], total_iters=CFG['warmup_epochs']
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[CFG['warmup_epochs']]
        )'''
        scheduler = main_lr_scheduler
        #loss_tr = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        loss_tr = LabelAwareSmoothing().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        best_answer = 0.0
        for epoch in range(CFG['epochs']):
            print(optimizer.param_groups[0]['lr'])
            train_one_epoch(epoch, model, model_ema, loss_tr, optimizer, train_loader, device, scheduler=scheduler,
                                schd_batch_update=False)
            answer = 0.0
            with torch.no_grad():
                answer = valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)
                if answer > best_answer:
                    torch.save(model.state_dict(), '{}_cutmix_fold_{}.pth'.format(CFG['model_arch'], fold))
                #if answer > 0.8:
                #    torch.save(model.state_dict(), '{}_epoch_{}_fold_{}_nolt.pth'.format(CFG['model_arch'], epoch, fold))
            if answer > best_answer:
                best_answer = answer

            # torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        print(best_answer)
        logger.info('BEST-TEST-ACC: ' + str(best_answer))
        torch.cuda.empty_cache()
