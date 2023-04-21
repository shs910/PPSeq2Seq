import random
import pandas as pd
import copy
from tqdm import tqdm

def split_data(src_path,tgt_dir,rate=0.9):
    with open(src_path,'r',encoding='utf-8') as f:
        with open(tgt_dir+'/train_2w.txt','w',encoding='utf-8') as train,\
            open(tgt_dir+'/valid_2w.txt','w',encoding='utf-8') as valid:
            lines=f.readlines()
            train_nums,train_pos,vlaid_num,valid_pos=0,0,0,0
            for line in lines:
                text,label=line.strip('\n').split('\t')
                if len(text)<40:
                    continue
                if random.random()<rate:
                    # train
                    train.write(line)
                    train_nums+=1
                    if label=='1':
                        train_pos+=1
                    if train_nums>50000:
                        break
                else:
                    # valid
                    valid.write(line)
                    vlaid_num+=1
                    if label=='1':
                        valid_pos+=1
            print('train positive {}/{}\nvalid positive {}/{}'.format(train_pos,train_nums\
                ,valid_pos,vlaid_num))

def merge_file(src_path_1,src_path_2,tgt_path):
    with open(src_path_1,'r',encoding='utf-8') as f1,\
        open(src_path_2,'r',encoding='utf-8') as f2:
        with open(tgt_path,'w',encoding='utf-8') as tgt:
            for line1 in f1.readlines():
                tgt.write(line1.strip('\n')+'\t'+'1\n')
            for line2 in f2.readlines():
                tgt.write(line2.strip('\n')+'\t'+'0\n')

def split_data_by_label(src_path,tgt_dir):
    with open(src_path,'r',encoding='utf-8') as src:
        with open(tgt_dir+'/oral.txt','w',encoding='utf-8') as oral,\
            open(tgt_dir+'/news.txt','w',encoding='utf-8') as news:
            count=0
            for line in src.readlines():
                text,label=line.strip('\n').split('\t')
                if label=='1':
                    oral.write(text+'\n')
                    count+=1
                elif label=='0':
                    news.write(text+'\n')
            print(count)

def sample_data(src_path,tgt_path,sample_num=1000,min_length=40,max_length=500):
    with open(src_path,'r',encoding='utf-8') as src:
        with open(tgt_path,'w',encoding='utf-8') as tgt:
            count_num=0
            lines=src.readlines()
            random.shuffle(lines)
            for line in lines:
                if len(line.strip('\n'))>min_length and len(line.strip('\n'))<max_length:
                    tgt.write(line)
                    count_num+=1
                    if count_num>=sample_num:
                        break
            
            print('sample num {}'.format(count_num))

def tocsv(src_path,tgt_path):
    with open(src_path,'r',encoding='utf-8') as src:
        lines=[]
        for lin in src.readlines():
            lin=lin.strip('\n')
            lines.append([lin,'####'])
        df=pd.DataFrame(lines,columns=['text','label'])
        df.to_csv(tgt_path,sep='\t',index=False)
def compute_avg_length(src_path):
    length=[]
    with open(src_path,'r',encoding='utf-8') as src:
        for line in src.readlines():
            length.append(len(line.strip('\n')))
    print(max(length),min(length),sum(length)/len(length))

def select_data_by_words(src_path,tgt_path,candidates_path):
    f=open(candidates_path,'r',encoding='utf-8')
    words=f.readlines()
    candidates=list(map(lambda x:x.strip('\n').lower(),words))
    print(candidates)
    count={}
    with open(src_path,'r',encoding='utf-8') as src:
        with open(tgt_path,'w',encoding='utf-8') as tgt:
            count_num=0
            for line in tqdm(src.readlines()):
                text=copy.deepcopy(line)
                line=line.lower()
                for words in candidates:
                    if words in line:
                        tgt.write(text)
                        count[words]=count.get(words,0)+1
                        count_num+=1
                        break
            print('total_num:{}'.format(count_num))
            count=sorted(count.items(),key=lambda x:x[1])
            print(count)

if __name__=='__main__':
    src_path='/home/hssun/sunhuashan/PPSeq2Seq/data/train_data/train_oral_informal.txt'
    tgt_path='/home/hssun/sunhuashan/PPSeq2Seq/data/train_data/train_news.txt'
    file_path='/home/hssun/sunhuashan/PPSeq2Seq/data/train_data/all_informal.txt'
    tgt_dir='/home/hssun/sunhuashan/PPSeq2Seq/data/train_data'
    # compute_avg_length(tgt_path)
    # split_data_by_label(src_path,tgt_path)
    # sample_data(src_path,tgt_path,sample_num=20915)
    # split_data(src_path,tgt_path,rate=0.9)
    # tocsv(src_path,tgt_path)
    # select_data_by_words(src_path,tgt_path,file_path)
    # merge_file(src_path,tgt_path,file_path)
    split_data(file_path,tgt_dir)
                
