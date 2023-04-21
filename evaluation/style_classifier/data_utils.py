'''
dataset dataloader etc.
'''

from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import torch
import random

class TextDataset(Dataset):
    def __init__(self,fil_path,mode='train',tokenizer=None,max_length=70) -> None:
        super(TextDataset,self).__init__()
        self.data=[]
        self.mode=mode
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.labels=[]
        #读取数据
        with open(fil_path,'r',encoding='utf-8') as f:
            lines=f.readlines()
            random.shuffle(lines)
            for line in lines:
                if self.mode=='test':
                    text=line.strip('\n')
                    self.data.append(text)
                else:
                    text,label=line.strip('\n').split('\t')
                    self.data.append(text)
                    self.labels.append(int(label))
    
    def __getitem__(self, index) :
        '''返回原始文本'''
        if self.mode=='test':
            return self.data[index]
        else:
            return {'text':self.data[index],'label': self.labels[index]}
    
    def __len__(self):
        return len(self.data)
    
    def collate_func(self,batch_data):
        if len(batch_data)==0:
            return {}
        if self.mode=='test':
            #batch_data就是list[str]
            labels=None
            encoding=self.tokenizer(batch_data,padding='max_length',truncation=True,max_length=self.max_length,return_tensors='pt')
            return {'encoding':encoding,'labels':labels}    
        else:
            #train or eval需要进一步整合
            text_list,label_list=[],[]
            for item in batch_data:
                text_list.append(item['text'])
                label_list.append(item['label'])
            encoding=self.tokenizer(text_list,padding='max_length',truncation=True,max_length=self.max_length,return_tensors='pt')
            labels=torch.tensor(label_list,dtype=torch.int64)
            return {'encoding':encoding,'labels':labels}
        
def create_dataloader(dataset, mode='train',batch_size=1,batchify_fn=None):
    #训练集打乱训练，验证集等不进行打乱
    shuffle = True if mode == 'train' else False
    return DataLoader(dataset,batch_size=batch_size,collate_fn=batchify_fn,shuffle=shuffle)

def split_data(label_path,src_path,tgt_dir,title,ref_path,pairs=False):
    with open(label_path,'r',encoding='utf-8') as label,\
            open(src_path,'r',encoding='utf-8') as src,\
            open(tgt_dir+'/'+title+'not_oral.txt','w',encoding='utf-8') as not_oral,\
                open(tgt_dir+'/'+title+'oral.txt','w',encoding='utf-8') as oral,\
                open(ref_path,'r',encoding='utf-8') as ref:
        c1,c2=0,0
        for line,label,ref in zip(src.readlines(),label.readlines(),ref.readlines()):
            if label.strip('\n')=='0':
                c1+=1
                if pairs:
                    not_oral.write(line.strip('\n')+'\t'+ref)
                else:
                    not_oral.write(line)
            elif label.strip('\n')=='1':
                if pairs:
                    oral.write(line.strip('\n')+'\t'+ref)
                else:
                    oral.write(line)
                c2+=1
            else:
                print('nothing')
        print('oral:{},not oral:{}'.format(c2,c1))

def select_data(label_path,src_path,tgt_path):
    with open(label_path,'r',encoding='utf-8') as label,\
            open(src_path,'r',encoding='utf-8') as src,\
            open(tgt_path,'w',encoding='utf-8') as tgt:
        for line,label in zip(src.readlines(),label.readlines()):
            if label.strip('\n')=='0':
                continue
            tgt.write(line)
    
if __name__=='__main__':
    label_path='/home/jcdu/zja/shs/evaluation/style_classifier/results/test_retell.oral.txt'
    src_path='/home/jcdu/zja/shs/evaluation/data/text.ref'
    tgt_path='/home/jcdu/zja/shs/evaluation/style_classifier/results/oral.ref.txt'
    select_data(label_path,src_path,tgt_path)
