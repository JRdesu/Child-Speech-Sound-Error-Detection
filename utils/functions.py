import torch
import torch.nn as nn
from torch.autograd import Variable  
import numpy as np
import pdb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os


# Ref:https://github.com/AzizCode92/Listen-Attend-and-Spell-Pytorch/blob/master/util/functions.py
# TimeDistributed function
# The goal is to apply same module on each timestep of every instance
# Input : module to be applied timestep-wise (e.g. nn.Linear)
#         3D input (sequencial) with shape [batch size, timestep, feature]
# output: Processed output      with shape [batch size, timestep, output feature dim of input module]
def TimeDistributed(input_module, input_x):
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    reshaped_x = input_x.contiguous().view(-1, input_x.size(-1))
    output_x = input_module(reshaped_x)
    return output_x.view(batch_size, time_steps,-1)

def CreateEmbedVariable(word_input, token_dim, embed_dim):
    embedding = nn.Embedding(token_dim, embed_dim)
    output_word = embedding(word_input.cpu()).view(word_input.size(0), 1, -1) # [B,1,V]
    return output_word # floattensor

def CreateOnehotVariable(word_input, encoding_dim=63):
    if type(input_x) is Variable:
        input_x = input_x.data 
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    input_x = input_x.unsqueeze(2)
    onehot_x = Variable(torch.FloatTensor(batch_size, time_steps, encoding_dim).zero_().scatter_(-1, input_x, 1)) #must be floattensor
    # 
    return onehot_x # floattensor

def PhoneErrorRate(pred, ground_truth, proc):
    batch_size = ground_truth.shape[0]
    dist = 0
    total = 0
    for idx in range(batch_size):
        pred_token = proc.decode(pred[idx])
        ref_token = proc.decode(ground_truth[idx])     
        dist = dist + levenshtein_dist(pred_token, ref_token)
        # pdb.set_trace()
        total = total + max(len(pred_token), len(ref_token))
    return dist/total

# def PhoneErrorRate_eval(pred, truth)
#     dist = levenshtein_dist(pred, truth)
    # total = max(len(pred), len(truth))
    # return dist/total


def levenshtein_dist(str1, str2):
    dp=np.zeros((len(str1)+1,len(str2)+1))
    m=len(str1)
    n=len(str2)
    for k in range(1,m+1):
        dp[k][0]=k
    for k in range(1,n+1):
        dp[0][k]=k
    for k in range(1,m+1):
        for j in range(1,n+1):
            dp[k][j]=min(dp[k-1][j],dp[k][j-1])+1 
            if str1[k-1]==str2[j-1]:
                dp[k][j]=min(dp[k][j],dp[k-1][j-1])
            else:
                dp[k][j]=min(dp[k][j],dp[k-1][j-1]+1)
    return dp[-1][-1]
'''
Reference:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''
def showAttention(input_tensor, y):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = sns.diverging_palette(128, 240, as_cmap=True)
    sns.heatmap(input_tensor.cpu().numpy(), annot=True, annot_kws={'size':6},
        fmt='.2f', cmap='Greens', linewidths=0.1, 
        ax=ax, yticklabels=y, square=True)
    ax.tick_params(axis='y', labelsize=10,labeltop=True)
    ax.tick_params(axis='x', labelsize=10, labelbottom=False, bottom= False, top=True, labeltop=True)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    

def showTrainingLoss(checkpoint, save_path):
    epoch = np.tile(np.array(checkpoint["epochs_count"]), 2)
    regions = [["train_losses"], ["valid_losses"]]
    region = []
    value = []
    for x in regions:
        region.extend(x*len(checkpoint[x[0]]))
        value.extend(checkpoint[x[0]])
# 

    region = np.array(region)
    value = np.array(value)
    loss = {}
    loss["Epoch"] = epoch
    loss['Dataset'] = region
    loss['Losses'] = value 
# 
    regions = [["train_scores"], ["valid_scores"]]
    region = []
    value = []
    for x in regions:
        region.extend(x*len(checkpoint[x[0]]))
        value.extend(checkpoint[x[0]])
# 

    region = np.array(region)
    value = np.array(value)
    score = {}
    score["Epoch"] = epoch
    score['Dataset'] = region
    score['Scores'] = value 
# 
    df1 = pd.DataFrame(data=loss, columns=["Epoch", "Dataset", "Losses"])
    df2 = pd.DataFrame(data=score, columns=["Epoch", "Dataset", "Scores"])
    fig_line, axes_line = plt.subplots(1,2, figsize=(20,10))
    sns.set_style("whitegrid", {'grid.linestyle' : '--'}) # whitegrid darkgrid dark white ticks
    # sns.axes_style() # get parameters
    # sns.despine(left=True) # remove left edge
    # sns.despine(offset=1, trim=True)
    sns.despine()
    colors = ["windows blue", "sunflower"]
    palette = sns.xkcd_palette(colors)
    sns.lineplot(data=df1, x="Epoch", y="Losses", hue="Dataset", 
    style='Dataset', palette=palette, markers=["o", "<"], ax=axes_line[0])
    sns.lineplot(data=df2, x="Epoch", y="Scores", hue="Dataset", 
    style='Dataset', palette=palette, markers=["o", "<"], ax=axes_line[1])
    for i in range(2):
        axes_line[i].xaxis.grid(False) # vertical gridline
        axes_line[i].yaxis.grid(True) # horizontal gridline

    plt.savefig(os.path.join(save_path, 'loss_per'), dpi=50)
    plt.show()

    # g = sns.FacetGrid(df1, col='Dataset')
    # g = g.map(plt.plot, "Epoch", "Losses", marker=".")
    # g = sns.FacetGrid(df2, col='Dataset')
    # g = g.map(plt.plot, "Epoch", "Scores", marker=".")
