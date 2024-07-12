import os, sys

import argparse
import pandas as pd
import cv2
import numpy as np
import torch
import sklearn 
import logging
import nibabel as nib
import sklearn.model_selection as skm
import torch.nn.functional as F

from config import *
from torch import nn 
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import deepsymnet as DeepSymNet

torch.cuda.empty_cache()

class CFG:
    seed = 1234
    image_path = IMG_PATH
    captions_path = CSV_PATH
    num_workers = 4
    image_lr = 1e-4
    weight_decay = 1e-5
    patience = 1
    factor = 0.8
    epochs = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    image_embedding = 72
    max_length = 200
    batch_size = 8

    logging_step = 10
    captions_path_log = SAVE_MODEL_LOG + '/log/'
    directory = 'best_model_fine_tune_' + str(datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
    tsbd_dir = os.path.join(captions_path_log, directory)
    os.makedirs(tsbd_dir)

    model_file_name = SAVE_MODEL_LOG 
    pretrained = True
    trainable = True
    temperature = 1.0

    size = 224
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1
    saved_best_model_name = str()

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

normLim = [0, 100]

def joinBrainNpy(leftArr, rightArr):
    fullBrain = np.vstack((np.flip(rightArr, axis=0), leftArr))

    return fullBrain       

def normVol(maskTmp):
        maskTmp[maskTmp < normLim[0]] = normLim[0]
        maskTmp[maskTmp > normLim[1]] = normLim[1]
        maskTmp = maskTmp - normLim[0]
        maskTmp = maskTmp / (normLim[1]-normLim[0])

        return maskTmp

def load_image(img_path):
    image_npy = np.load(img_path, allow_pickle=True)
    leftBrain = image_npy.item().get('leftBrain')
    rightBrain = image_npy.item().get('rightBrain')
    rightBrain = np.flip(rightBrain, axis=0)

    maskedLeft = leftBrain.copy()
    maskedRight = rightBrain.copy()

    full_brain = joinBrainNpy(maskedLeft, maskedRight)
    
    maskedLeft = normVol(maskedLeft)
    maskedRight = normVol(maskedRight)

    image = np.expand_dims(full_brain, axis=0)
    maskedLeft = np.expand_dims(maskedLeft, axis=0)
    maskedRight = np.expand_dims(maskedRight, axis=0)

    return maskedLeft.astype(np.float32), maskedRight.astype(np.float32), image


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, label, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.label = list(label)
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {}
    
        imageL, imageR, image = load_image(f"{CFG.image_path}/{self.image_filenames[idx]}")
    
        imageL, imageR, image = np.float32(imageL), np.float32(imageR), np.float32(image)
        item['imageL'] = torch.from_numpy(imageL.copy()) 
        item['imageR'] = torch.from_numpy(imageR.copy()) 
        item['image'] = torch.from_numpy(image.copy())          
        item['label'] = self.label[idx]
        return item

    def __len__(self):
        return len(self.label)

class testDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, label, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.label = list(label)
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {}
    
        imageL, imageR, image = load_image(f"{CFG.image_path}/{self.image_filenames[idx]}")
        fname = self.image_filenames[idx]

        imageL, imageR, image = np.float32(imageL), np.float32(imageR), np.float32(image)
        item['imageL'] = torch.from_numpy(imageL.copy()) 
        item['imageR'] = torch.from_numpy(imageR.copy())
        item['image'] = torch.from_numpy(image.copy())           
        item['label'] = self.label[idx]
        return item, fname

    def __len__(self):
        return len(self.label)

def load_csv(index, ti):

    testdf = pd.read_csv(f'{CFG.captions_path}test_{str(index)}_{str(ti)}.csv')

   
    return testdf

def build_loaders(dataframe, doShuffle=False):
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["label"].values,
        transforms=None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=doShuffle,
    )
    return dataloader

def build_testloaders(dataframe, doShuffle=False):
    dataset = testDataset(
        dataframe["image"].values,
        dataframe["label"].values,
        transforms=None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=doShuffle,
    )
    return dataloader


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(ft_begin_index))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.fc1 = nn.Linear(256, 1) 
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        z = torch.flatten(x, 1)
        z = self.fc1(z)
        return x, z

class CLIPModel(nn.Module):
    def __init__(
        self,
        image_embedding=CFG.image_embedding,
        temperature=CFG.temperature,
    ):
        super().__init__()

        self.image_encoder = DeepSymNet.torch_sNetAutoPreprocessingVggNetWithSkipConn(n_classes=1,
                                                            depthBefore=3, 
                                                            depthAfter=2, 
                                                            nFilters=24,  
                                                            nConv=3,
                                                            addDenseLayerNeurons=15,
                                                            last_fc=False
                                                            ).to(CFG.device)

        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        imL = batch['image'][:,:,:73,:,:]
        imR = batch['image'][:,:,73:,:,:]
    
        image_features = self.image_encoder(imL, imR)
        image_embeddings, image_logit = self.image_projection(image_features)
        return image_logit

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = True

def round_nd_array(array):
        return [round(val, 4) for val in array]


class EarlyStopping():
    def __init__(self, tolerance):
     
        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.last_auc = 0.
    
    def __call__(self, val_auc):
        if val_auc <= self.last_auc:
            self.counter += 1
            print(f'With count {self.counter} at auc {val_auc}')
            if self.counter >= self.tolerance:
                self.early_stop = True
        elif val_auc > self.last_auc:
            self.last_auc = val_auc

# %%
def main(ti, num_fold):
    print(CFG.device)
    currPredLogitLst = []
    currTgtLst = []
    currPredLst = []
    currProbLst = []
    test_fname = []
    for n in range(num_fold):
        test_df = load_csv(n, ti)
        # train_loader = build_loaders(train_df, doShuffle=True)
        test_loader = build_testloaders(test_df, doShuffle=False)

        model = CLIPModel().to(CFG.device)
        model.load_state_dict(torch.load(SAVED_PRETRAINED_MODEL, map_location=CFG.device), strict=False)
        set_parameter_requires_grad(model, CFG.trainable)

        #==================== Testing
        print('Testing')

        model = CLIPModel().to(CFG.device)

        saved_best_dir = f'{CFG.model_file_name}/iteration_{ti}/fold_{n}/'
        CFG.saved_best_model_name = f'{saved_best_dir}{os.listdir(saved_best_dir)[-1]}'
        model.load_state_dict(torch.load(CFG.saved_best_model_name, map_location=CFG.device))
        print(f'Loaded the best model: {CFG.saved_best_model_name}')
        model.eval()

        loopinTgtLst = []
        loopinProbLst = []
        with torch.no_grad():
            for i, (batch, fname) in enumerate(tqdm(test_loader)):
                batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
                image_logit = model(batch)
                i_logit = image_logit.cpu()

                probs = torch.sigmoid(i_logit)
                predicted_vals = probs > 0.5
                
                test_targets = batch["label"].to(CFG.device)
                test_targets = test_targets.unsqueeze(1)

                loopinTgtLst = np.append(loopinTgtLst, test_targets.clone().flatten().tolist())
                loopinProbLst = np.append(loopinProbLst, probs.clone().flatten().tolist())

                currTgtLst = np.append(currTgtLst, test_targets.clone().flatten().tolist())
                currProbLst = np.append(currProbLst, probs.clone().flatten().tolist())
                for item in fname:
                    test_fname.append(item)
                # for i in range(len(fname)):
                #     print(fname[i], batch['label'][i], probs.clone().flatten().tolist()[i])
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(loopinTgtLst, loopinProbLst, pos_label=1)
        loopinAUC = round(sklearn.metrics.auc(fpr, tpr), 4)

        lplength = len(loopinTgtLst)
        lplength_pred = len(loopinProbLst)

    list_dic = {'image': test_fname, 'pred': currProbLst, 'label': currTgtLst}
    df = pd.DataFrame(list_dic)
    df.to_csv(f'{OUTPUT_PROB_CSV}/auc_5fold_{ti}_times.csv', index=False)
    
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(currTgtLst, currProbLst, pos_label=1)
    aucValSet = round(sklearn.metrics.auc(fpr, tpr), 4)

    length = len(currTgtLst)
    length_pred = len(currProbLst)
    print(f'Test set AUC={aucValSet} with length {length} and {length_pred}') 
    #==================== 
    
if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--ti')
    parser.add_argument('--fold')
    args = parser.parse_args()
    # Start
    main(args.ti, args.fold)

