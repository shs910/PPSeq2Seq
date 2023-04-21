from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import torch

class TextDataset(Dataset):
    def __init__(self,texts,labels,tokenizer) -> None:
        super(TextDataset,self).__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer=tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return {'text':self.texts[index],'label':self.labels[index]}
    
    def collate_fn(self,bath_data):
        if len(bath_data)==0:
            return {}
        bath_text,bath_labels=[],[]
        for item in bath_data:
            bath_text.append("paraphrase: "+item['text']+" </s>")
            bath_labels.append(int(item['label']))
        #对文本进行编码
        if self.tokenizer.model_max_length<70:
            inputs=self.tokenizer(bath_text,padding=True,truncation=True,max_length=self.tokenizer.max_model_length-3,return_tensors='pt')
        else:
            inputs=self.tokenizer(bath_text,padding='longest',truncation=True,return_tensors='pt')
        labels=torch.tensor(bath_labels,dtype=torch.int64)
        return {'inputs':inputs,'labels':labels}
    
def get_dataloader(dataset,batch_size,mode='train'):
    shuffle=False
    if mode=='train':
        shuffle=True
    return DataLoader(dataset=dataset,\
        batch_size=batch_size,shuffle=shuffle,collate_fn=dataset.collate_fn)

def loaddata(data_path,mode='train',sep='\t'):
    texts,labels=[],[]
    print('load {} data'.format(mode))
    with open(data_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
        sorted_lines=sorted(lines,key=len)
        print('sorted!!')
        for line in tqdm(sorted_lines):
            line=line.strip('\n').split(sep)
            texts.append(line[0])
            labels.append(line[1])
    return texts,labels
    