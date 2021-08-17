'''python3 test_ner.py  --initialization glove --char_embeddings 1  --layer_normalization 1  --crf  1  --model_file trained_models_ner/part_2.2_crf_glove_char_ln.pth --test_data_file ../data/ner_test_data.txt --output_file trained_models_ner/predictions_part_2.2_crf_glove_char_ln.txt --glove_embeddings_file ../data/glove/glove.6B.100d.txt --vocabulary_input_file trained_models_ner/part_2.2_crf_glove_char_ln.vocab '''
from collections import Counter
import itertools
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import re
from torch.utils.data import Dataset
import torch
import time
import copy
# from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from seqeval.metrics import classification_report, f1_score, accuracy_score
import os
import random
import logging
import argparse
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# train_ner.py --initialization [random | glove ] --char_embeddings [ 0 | 1 ] --layer_normalization [ 0 | 1 ] --crf [ 0 | 1 ] --output_file <path to the trained model> --data_dir <directory containing data> --glove_embeddings_file <path to file containing glove embeddings> --vocabulary_output_file <path to the file in which vocabulary will be written>

parser = argparse.ArgumentParser(description='COL 870 Assignment 1.2')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size to use')
parser.add_argument('--epochs', default=100, type=int, help='num of epochs to train')
parser.add_argument('--seed', type = int, default = 870, help='random seed')
parser.add_argument('--patience', type=int, default = 100, help='patience for seed')
parser.add_argument('--char_embeddings',type=int,default=0,help='use character embeddings')
parser.add_argument('--initialization',type=str,default='random',help='[random|glove]')
parser.add_argument('--model_file',type=str,default='trained_models_ner/part_2.2_crf_glove_char_ln.pth',help='path to model')

parser.add_argument('--output_file',type=str,default='file.txt',help='pathtext')
parser.add_argument('--test_data_file',type=str,default=' ../data/ner_test_data.txt',help='path to test data')
parser.add_argument('--glove_embeddings_file',type=str,default='glove/glove.6B.100d.txt',help='path to glove embeddings')
parser.add_argument('--vocabulary_input_file',type=str,default='words.vocab',help='path to vocab')
parser.add_argument('--crf',type=int,default=0,help='use crf')
parser.add_argument('--layer_normalization',type=int,default=0,help='use ln')


args = parser.parse_args()
print('epochs?',args.epochs)
print('batch_size?',args.batch_size)
print('seed?',args.seed)
print('patience?',args.patience)
print('initialization?',args.initialization)
print('char?',args.char_embeddings==1)
print('ln?',args.layer_normalization==1)
print('crf?',args.crf==1)




def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed = args.seed)

def filereader(file_path):
  sentences=[]
  gold=[]
  new_sentence=[]
  new_gold=[]
  file = open(file_path,'r')
  for line in file.readlines():
    a=line.split()
    if(len(a)==0):
      sentences.append(new_sentence)
      gold.append(new_gold)
      new_sentence=[]
      new_gold=[]

    else:
      new_sentence.append(a[0])
      new_gold.append(a[3])

  file.close()
  return sentences[1:], gold[1:]

# def get_counts_vocab(sentences,gold):
#   words = {}
#   labels = {}
#   characters = {}
#   for s,g in zip(sentences,gold):
#     for w,l in zip(s,g):
#       if w not in words:
#         words[w]=0
#       words[w]+=1
#       if l not in labels:
#         labels[l]=0
#       labels[l]+=1

#       for c in list(w):
#         if c not in characters:
#           characters[c]=0
#         characters[c]+=1
#   return words,characters,labels
# def true_vocab(word_count_vocab,char_count_vocab,label_count_vocab,min_wf,min_cf):
#   word_vocab={'<unk>':0,'<pad>':1}
#   char_vocab={'<unk>':0,'<pad>':1}
#   label_vocab={}
#   i=2
#   for k in word_count_vocab:
#     if word_count_vocab[k]>=min_wf:
#       word_vocab[k]=i
#       i+=1
#   i=2
#   for k in char_count_vocab:
#     if char_count_vocab[k]>=min_cf:
#       char_vocab[k]=i
#       i+=1
#   wts=[]
#   i=0
#   for k in label_count_vocab:
#     label_vocab[k]=i
#     i+=1
#     wts.append(1/label_count_vocab[k])
#   return word_vocab, char_vocab,label_vocab,wts
# sentences,gold = filereader(args.data_dir+'/train.txt')
# wcv,ccv,lcv = get_counts_vocab(sentences,gold)
# wv,cv,lv,wts = true_vocab(wcv,ccv,lcv,3,2)
# np.save(args.vocabulary_output_file+'words.npy', wv) 
# np.save(args.vocabulary_output_file+'char.npy', cv) 
# np.save(args.vocabulary_output_file+'labels.npy', lv) 
wv = np.load(args.vocabulary_input_file+'words.npy',allow_pickle='TRUE').item()
cv = np.load(args.vocabulary_input_file+'char.npy',allow_pickle='TRUE').item()
lv = np.load(args.vocabulary_input_file+'labels.npy',allow_pickle='TRUE').item()
def get_tensors(file,wv,cv,lv):
  sentences,gold = filereader(file)
  max_wl = 0
  max_cl = 0
  for s in sentences:
    if max_wl<len(s):
      max_wl=len(s)
    for w in s:
      if max_cl<len(w):
        max_cl=len(w)
  print('Maximum word length:',max_wl,'Maximum char length:',max_cl)
  s_list = []
  c_list = []
  t_list = []
  seq_len = []
  char_len = []
  for s,g in tqdm(zip(sentences,gold)):
    sl = []
    cl = []
    tl = []
    seq_len.append(len(s))
    word_char_len=[]
    
    i=0

    for w,l in zip(s,g):
      word_char_len.append(len(w))
      ccl = []
      j=0
      for c in list(w):
        ccl.append(cv.get(c,cv['<unk>']))
        j+=1
      while j<max_cl:
        ccl.append(cv['<pad>'])
        j+=1

      cl.append(ccl)
      sl.append(wv.get(w,wv['<unk>']))
      tl.append(lv[l])
      i+=1
    while i<max_wl:
      word_char_len.append(0)
      sl.append(wv['<pad>'])
      cl.append([cv['<pad>']]*max_cl)
      tl.append(lv['O'])
      i+=1
    s_list.append(sl)
    c_list.append(cl)
    t_list.append(tl)
    char_len.append(word_char_len)
  return torch.LongTensor(s_list),torch.LongTensor(c_list),torch.LongTensor(t_list),torch.LongTensor(seq_len), torch.LongTensor(char_len)

    

class MyDataset(Dataset):
  def __init__(self, tensors):
    self.s_list = tensors[0]
    self.c_list = tensors[1]
    self.t_list = tensors[2]
    self.seq_len = tensors[3]
    self.char_len = tensors[4]
    self.data_size = self.s_list.shape[0]
    print('size:',self.data_size)

  def __getitem__(self, i):
      return self.s_list[i], self.c_list[i], self.t_list[i], self.seq_len[i], self.char_len[i]

  def __len__(self):
      return self.data_size

# train_data = MyDataset(get_tensors(args.data_dir+'/train.txt',wv,cv,lv))
# val_data = MyDataset(get_tensors(args.data_dir+'/dev.txt',wv,cv,lv))
test_data = MyDataset(get_tensors(args.test_data_file,wv,cv,lv))
dataloaders = {}

# dataloaders['train'] = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size, shuffle=True, num_workers=4)
# dataloaders['val'] = torch.utils.data.DataLoader(val_data,batch_size=args.batch_size, shuffle=False, num_workers=4)
dataloaders['test'] = torch.utils.data.DataLoader(test_data,batch_size=args.batch_size, shuffle=False,num_workers=4)

import numpy as np

print('Indexing word vectors.')

embeddings_index = {}
f = open(args.glove_embeddings_file, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

import copy

word_vocab_embeddings = {}

for k, v in wv.items():
  try:
    if v not in word_vocab_embeddings:
      word_vocab_embeddings[v] = embeddings_index[k] # fixes init of known words
  except KeyError:
    if v not in word_vocab_embeddings:
      word_vocab_embeddings[v] = np.random.normal(scale = 0.6, size = (100,)) # change to change init of unknown words
wt_matrix = torch.zeros(len(wv),100)
for i in range(len(wv)):
  wt_matrix[i]=torch.Tensor(word_vocab_embeddings[i])

class MyCRF(nn.Module):
  def __init__(self,tagset_size):
    super().__init__()
    self.tagset_size=tagset_size
    self.st = nn.Parameter(torch.FloatTensor(tagset_size).uniform_(-0.1, 0.1))
    self.et = nn.Parameter(torch.FloatTensor(tagset_size).uniform_(-0.1, 0.1))
    self.t = nn.Parameter(torch.FloatTensor(tagset_size,tagset_size).uniform_(-0.1, 0.1))

  def forward(self,e,tags,mask):
    e = e.transpose(0,1)
    tags = tags.transpose(0,1)
    mask = mask.transpose(0,1)

    sl,bs = tags.shape
    idx = torch.arange(bs)

    score = self.st[tags[0]]
    score+= e[0,idx,tags[0]]
    normalize = self.st+e[0]

    for i in range(sl-1):
      score+=self.t[tags[i],tags[i+1]]*mask[i+1]
      score+=e[i+1,idx,tags[i+1]]*mask[i+1]

      ns = torch.logsumexp(normalize.unsqueeze(2)+self.t+e[i+1].unsqueeze(1),dim=1)

      normalize = torch.where(mask[i+1].unsqueeze(1),ns,normalize)
    
    score+=self.et[tags[mask.sum(dim=0)-1,idx]]
    normalize+=self.et
    normalize = torch.logsumexp(normalize,dim=1)

    return (score-normalize).sum()/ mask.sum()

  def decode(self,e,mask):
    e=e.transpose(0,1)
    mask=mask.transpose(0,1)
    sl,bs = mask.shape
    score = self.st+e[0]
    tlist = []
    for i in range(sl-1):
      ns,indices = (score.unsqueeze(2)+self.t+e[i+1].unsqueeze(1)).max(dim=1)
      tlist.append(indices)
      score=torch.where(mask[i+1].unsqueeze(1),ns,score)

    score += self.et

    lengths = mask.sum(dim=0)-1
    predictions=[]
    for i in range(bs):
      _,a=score[i].max(dim=0)
      bt= [a.item()]

      for j in reversed(tlist[:lengths[i]]):
        bt.append(j[i][bt[-1]].item())
      bt.reverse()
      predictions.append(bt)
    
    return predictions



class myLayerNorm(nn.Module):
  def __init__(self,hidden_dim: int,epsilon=1e-5):
    super(myLayerNorm,self).__init__()
    self.hidden_dim = hidden_dim
    self.beta = nn.Parameter(torch.zeros(hidden_dim))
    self.gamma = nn.Parameter(torch.ones(hidden_dim))
    self.epsilon=epsilon
  def forward(self,x):
    bsz,h =  x.shape
    mean = torch.mean(x,dim=1)
    biased_variance = torch.var(x,dim=1,unbiased=False)

    normalized_x = x-mean.reshape(bsz,1)
    normalized_x = normalized_x / torch.sqrt(self.epsilon+biased_variance.reshape(bsz,1))
    
    return self.beta.reshape(1,self.hidden_dim)+ self.gamma.reshape(1,self.hidden_dim)*normalized_x

    

class myLayerNormLSTMCell(nn.Module):


  def __init__(self,inp_size,hidden_size): 
    super(myLayerNormLSTMCell, self).__init__()


    self.gates_x= nn.Linear(inp_size, 4*hidden_size, bias=True)
    self.gates_h = nn.Linear(hidden_size, 4*hidden_size, bias=True)

    self.ln_gates_x = myLayerNorm(4*hidden_size)
    self.ln_gates_h = myLayerNorm(4*hidden_size)

    self.ln_cell = myLayerNorm(hidden_size)


  def forward(self,x,h,c):
    gx = self.ln_gates_x(self.gates_x(x))
    gh = self.ln_gates_h(self.gates_h(h))
    g = gx+gh

    ig, fg, og, input = g.chunk(4, dim= 1)
    ig = F.sigmoid(ig)
    fg = F.sigmoid(fg)
    og = F.sigmoid(og)
    input = F.tanh(input)

    c_new = c*fg+input*ig
    h_new = og*F.tanh(self.ln_cell(c_new))
    return h_new, (h_new, c_new)
# nn.LSTM(wed+ced*2,(wed+ced*2)//2,num_layers=2,bidirectional=True)

class LNLSTM(nn.Module):
  def __init__(self,input_size,hidden_size,num_layers=1,bidirectional = True):
    super(LNLSTM,self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bidirectional = bidirectional
    if bidirectional:
      self.d = 2
    else:
      self.d = 1

    layers_forward = []

    size = input_size

    for i in range(num_layers):
      layers_forward.append(myLayerNormLSTMCell(size,hidden_size))
      size = hidden_size
    self.lf = nn.ModuleList(layers_forward)
    if bidirectional:
      size = input_size
      layers_backward = []
      for i in range(num_layers):
        layers_backward.append(myLayerNormLSTMCell(size,hidden_size))
        size = hidden_size
      self.lb = nn.ModuleList(layers_backward)
  def forward(self, x):
      # x shape--> sl,bsz,emb
      h0 = torch.zeros(self.d,self.num_layers,x.shape[1],self.hidden_size).float().to(x.device)
      c0 = torch.zeros(self.d,self.num_layers,x.shape[1],self.hidden_size).float().to(x.device)
      # for each layer, store the last hidden and cell state
      f_h=[]
      f_c=[]
      f_input = x
      for i in range(len(self.lf)):
        h = h0[0][i] # bsz,hsz
        c = c0[0][i] # bsz,hsz
        out = []
        for j in range(f_input.shape[0]):
          o,(h,c) = self.lf[i](f_input[j],h,c) #o shape--> bsz,hsz
          out.append(o.unsqueeze(0))# 1,bsz,hsz
        f_h.append(h.unsqueeze(0)) # h at final time step shape--> 1,bsz,hsz
        f_c.append(c.unsqueeze(0)) # c at final time step shape--> 1,bsz,hsz
        f_input = torch.cat(out,dim=0) # shape sl,bsz,hsz
      f_h = torch.cat(f_h,dim=0)# shape nl,bsz,hsz
      f_c = torch.cat(f_c,dim=0)# shape nl,bsz,hsz
      
      if self.bidirectional:
       
        b_h=[]
        b_c=[]
        b_input = x
        # [a,b,c,d]
        for i in range(len(self.lb)):
          h = h0[1][i]
          c = c0[1][i]
          out = []
          sl = b_input.shape[0]
          for j in range(sl):
            o,(h,c) = self.lb[i](b_input[sl-j-1],h,c)
            out.append(o.unsqueeze(0))
          b_h.append(h.unsqueeze(0))
          b_c.append(c.unsqueeze(0))
          out.reverse()
          # [a_o,b_o,c_o,d_o]
          b_input = torch.cat(out,dim=0) #sl,bsz,hsz in correct original order
        b_h = torch.cat(b_h,dim=0)
        b_c = torch.cat(b_c,dim=0)
      
        lstm_out = torch.cat([f_input,b_input],dim=2) #sl,bsz,hsz*2
        lstm_h = (f_h,b_h)
        lstm_c = (f_c,b_c)
      else:
        lstm_out = f_input
        lstm_h = f_h
        lstm_c = f_c
      return lstm_out, (lstm_h,lstm_c)
        
class MyModel(nn.Module):
  def __init__(self,wed,ced,nc,wv_len,cv_len,glove_matrix,use_character=True,use_crf=True,use_glove=True,use_ln=True):
    super(MyModel,self).__init__()
    self.wed = wed
    self.ced = ced
    self.use_character=use_character
    self.use_crf = use_crf
    if use_character:
      self.cv_len = cv_len
      self.c_embedding = nn.Embedding(cv_len,ced)
      if use_ln:
        self.c_lstm = LNLSTM(ced,ced,num_layers=1,bidirectional=True)
        self.w_bilstm = LNLSTM(wed+ced*2,(wed+ced*2)//2,num_layers=2,bidirectional=True)
      else:
        self.c_lstm = nn.LSTM(ced,ced,num_layers=1,bidirectional=True)
        self.w_bilstm = nn.LSTM(wed+ced*2,(wed+ced*2)//2,num_layers=2,bidirectional=True)
      self.final = nn.Sequential(nn.Linear(wed+ced*2,256),nn.ReLU(),nn.Linear(256,nc))
      # self.c_net = nn.Sequential(nn.Linear(ced*2,ced*2),nn.Tanh())
    else:
      if use_ln:
        self.w_bilstm = LNLSTM(wed,wed//2,num_layers=2,bidirectional=True)
      else:
        self.w_bilstm = nn.LSTM(wed,wed//2,num_layers=2,bidirectional=True)
      self.final = nn.Sequential(nn.Linear(wed,256),nn.ReLU(),nn.Linear(256,nc))



    self.nc = nc
    self.wv_len = wv_len
    self.w_embedding  = nn.Embedding(wv_len,wed)
    if use_glove:
      assert wed==100
      self.w_embedding.load_state_dict({'weight': glove_matrix})
      # from_pretrained(glove_matrix)
      
    self.dropout = nn.Dropout(0.5)

    if use_crf:
      self.crf=MyCRF(nc)

      

  def forward(self,s,c,t,sl):
    e_s = self.w_embedding(s)
    e_s = self.dropout(e_s)
    if self.use_character:
    
      bsz,seq_len,c_len = c.shape
      c=c.reshape(bsz*seq_len,c_len)
      # bsz*seq_len,ced

      # print(s)
  


      e_c = self.c_embedding(c)
      e_c = self.dropout(e_c)
      e_c = e_c.permute(1,0,2)
      
      # print(e_c)
      _,(e_c,_) = self.c_lstm(e_c)
      e_c = torch.cat([e_c[0],e_c[1]],dim=1)
      e_c = e_c.reshape(bsz,seq_len,2*self.ced)
      # e_c = e_c.permute(1,0,2)
      # e_c  = self.c_net(e_c)
      e_s = torch.cat([e_s,e_c],dim=2)

    e_s = e_s.permute(1,0,2)
    f,_ = self.w_bilstm(e_s)
    f = f.permute(1,0,2)
    f = self.final(f)
    if self.use_crf:
      m = torch.arange(t.shape[1]).to(device).expand(len(sl), t.shape[1]) < sl.unsqueeze(1)
      if self.training:
        scores= self.crf(f,t,m)
      else:
        scores = 0
      predictions = self.crf.decode(f,m)
      return -scores,predictions

    else:
      
      return f

# net = MyModel(100,25,len(lv),len(wv),len(cv),wt_matrix,use_character=(args.char_embeddings==1),use_crf=(args.crf==1),use_glove=(args.initialization=='glove'),use_ln=(args.layer_normalization==1)).to(device)
net = torch.load(args.model_file).to(device)
inverse_wv = {v: k for k,v in wv.items()}
inverse_lv = {v: k for k,v in lv.items()}

def invert(pred,true):
  f_true=[]
  f_pred=[]
  for p,t in zip(pred,true):
    tr = []
    for gt in t:
      tr.append(inverse_lv[gt])
    f_true.append(tr)
    pr = []
    for gp in p:
      pr.append(inverse_lv[gp])
    f_pred.append(pr)
  return f_pred,f_true
def evaluate(model,use_crf=True):
  true=[]
  pred=[]
  model.eval()
  for i,(s,c,t,sl,cl) in tqdm(enumerate(dataloaders['test'])):
    max_word_len = sl.max()
    max_char_len = cl.max()
    s=s[:,:max_word_len]
    c=c[:,:max_word_len,:max_char_len]

    t=t[:,:max_word_len]
    cl=cl[:,:max_word_len]

    s=s.to(device)
    c=c.to(device)
    t=t.to(device)
    sl=sl.to(device)
    if use_crf:
      _,predictions = model(s.to(device),c.to(device),0*t.to(device),sl.to(device))
      for i,a in enumerate(sl):
        pred.append(predictions[i][:a])
        true.append(t[i][:a].tolist())
    else:
      logits = model(s.to(device),c.to(device),0*t.to(device),sl.to(device))
      predictions = logits.argmax(dim=2)
      for i,a in enumerate(sl):
        pred.append(logits[i][:a].argmax(dim=1).tolist())
        true.append(t[i][:a].tolist())
    
  f_pred,f_true=invert(pred,true)
  return f_true,f_pred



f_true,f_pred = evaluate(net,use_crf=(args.crf==1))

# f=open(args.output_file,'w')
# for l in f_pred:
#     f.write(' \n')
#     for a in l:
#         f.writeline(a+'\n')
# f.close()

def test_output_write(read_path,write_path,f_pred):
  sentences=[]
  new_sentence=[]
  i=-1
  j=-1
  file = open(read_path,'r')
  for line in file.readlines():
    a=line.split()
    
    
    if(len(a)==0):
      j+=1
      i=-1
      sentences.append(new_sentence)
      new_sentence=[]

    else:
      i+=1
      new_sentence.append([a[0],a[1],a[2],f_pred[j][i]])
  sentences=sentences[1:]
  file.close()
  file_w = open(write_path,'w')
  for l in sentences:
    file_w.write(' \n')
    for a in l:
        file_w.write(a[0]+' '+a[1]+' '+a[2]+' '+a[3]+'\n')

test_output_write(args.test_data_file,args.output_file,f_pred)
  