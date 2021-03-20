import os
from os.path import join as oj
import sys
sys.path.append('../src')
import numpy as np
import seaborn as sns
import torch
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from functools import partial

import data
import config
import features
import train_reg


# we generally ignore the two pi4p datasets since their outcome is substantially different
print("loading data")
dsets = ['clath_aux+gak_a7d2', 'clath_aux+gak', 'clath_aux+gak_a7d2_new', 'clath_aux+gak_new', 'clath_gak', 'clath_aux_dynamin'] # need to add dynamin
splits = ['train', 'test']

dfs = []
for dset in tqdm(dsets):
    df = data.get_data(dset)

    # add dset
    df['dset'] = dset    
    
    # add split
    df['split'] = 'train'
    df['split'][df.cell_num.isin(config.DSETS[dset]['test'])] = 'test'
    
    # deal with short, long, hotspots
    df['short'] = df.lifetime <= 15
    df['hotspots'] = df['hotspots'] | df['long'] | df.lifetime > 150
    
    # add same_length_track
    df['X_same_length'] = df['X'].apply(features.downsample)
    
    # add outcomes Y_sig_mean and Y_sig_mean_normalized
    df = train_reg.add_sig_mean(df, resp_tracks=['Y'])     
    
    dfs.append(df)
    
df_full = pd.concat(dfs)
cols_to_drop = ['X_starts', 'X_extended', 'X_ends',
                'Y_starts', 'Y_ends', 'sig_idxs', 'catIdx',
                'Z_starts', 'Z_ends', 'valid', 'y_z_score']

df_full = df_full.drop(columns=cols_to_drop)
df_full.to_pickle(oj(config.DIR_PROCESSED, 'df_full.pkl'))