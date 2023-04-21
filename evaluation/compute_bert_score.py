from bert_score import BERTScorer
from tqdm import tqdm
import copy
import numpy as np

def load_data(ref_path,hyp_path,batch_size=64):
    with open(ref_path,'r',encoding='utf-8') as ref:
        ref_data=[]
        ref_bath_data=[]
        count=0
        for line in ref.readlines():
            ref_bath_data.append(line.strip('\n'))
            count+=1
            if count>=batch_size:
                ref_data.append(copy.deepcopy(ref_bath_data))
                ref_bath_data.clear()
                count=0
        if ref_bath_data:
            ref_data.append(copy.deepcopy(ref_bath_data))

    with open(hyp_path,'r',encoding='utf-8') as hyp:
        hyp_data=[]
        hyp_bath_data=[]
        count=0
        for line in hyp.readlines():
            hyp_bath_data.append(line.strip('\n'))
            count+=1
            if count>=batch_size:
                hyp_data.append(copy.deepcopy(hyp_bath_data))
                hyp_bath_data.clear()
                count=0
        if hyp_bath_data:
            hyp_data.append(copy.deepcopy(hyp_bath_data))
    assert len(hyp_data)==len(ref_data)
    print('batch_size: {},batch_nums: {}'.format(batch_size,len(hyp_data)))
    return ref_data,hyp_data
    
def load_data_onefile(src_path,batch_size):
    with open(src_path,'r',encoding='utf-8') as src:
        hyp_data=[]
        hyp_bath_data=[]
        ref_data=[]
        ref_bath_data=[]
        count=0
        for line in src.readlines():
            hyp,ref=line.strip('\n').split('\t')
            hyp_bath_data.append(hyp)
            ref_bath_data.append(ref)
            count+=1
            if count>=batch_size:
                hyp_data.append(copy.deepcopy(hyp_bath_data))
                ref_data.append(copy.deepcopy(ref_bath_data))
                hyp_bath_data.clear()
                ref_bath_data.clear()
                count=0
        if hyp_bath_data:
            hyp_data.append(copy.deepcopy(hyp_bath_data))
            ref_data.append(copy.deepcopy(ref_bath_data))
    assert len(hyp_data)==len(ref_data)
    print('batch_size: {},batch_nums: {}'.format(batch_size,len(hyp_data)))
    return ref_data,hyp_data

def compute(src_path,batch_size,one_file=True,hyp_path=None,title='withoutClassifier'):
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    if one_file:
        ref_data,hyp_data=load_data_onefile(src_path,batch_size)
    else:
        ref_data,hyp_data=load_data(ref_path=src_path,hyp_path=hyp_path,batch_size=batch_size)
    f1,r,p=[],[],[]
    for ref,hyp in zip(tqdm(ref_data),hyp_data):
        P, R, F1 = scorer.score(hyp, ref)
        f1.append(F1.mean().item())
        r.append(R.mean().item())
        p.append(P.mean().item())
    print(title)
    print('f1:{},recall:{},precise:{}'.format(\
        round(np.mean(f1),4),round(np.mean(r),4),round(np.mean(p),4)))

if __name__=='__main__':
    src_path='/home/hssun/sunhuashan/PPSeq2Seq/evaluation/test_data.txt'
    hyp_path='/home/hssun/sunhuashan/PPSeq2Seq/inference/res/t5_1000_loss_dividwithoutClassifier.txt'
    ref_path='/home/hssun/sunhuashan/PPSeq2Seq/inference/res/news_1000.txt'
    batch_size=64*2
    title='withClassifier'
    # compute(ref_path,batch_size,title=title,hyp_path=hyp_path,one_file=False)
    compute(src_path,batch_size,one_file=True,title='title')

        


