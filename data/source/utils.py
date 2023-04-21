import os
def split_data(src_path,tgt_dir):
    with open(src_path,'r',encoding='utf-8') as src:
        informal_path=os.path.join(tgt_dir,'informal_words.txt')
        formal_path=os.path.join(tgt_dir,'formal_words.txt')
        with open(informal_path,'w',encoding='utf-8') as informal:
            with open(formal_path,'w',encoding='utf-8') as formal:
                informal_words,formal_words=[],[]
                for line in src.readlines():
                    w1,w2=line.strip('\n').split('--')
                    informal_words.append(w1+'\n')
                    formal_words.append(w2+'\n')
                print(len(informal_words))
                informal_words=list(set(informal_words))
                print(len(informal_words))
                formal_words=list(set(formal_words))
                for w1,w2 in zip(informal_words,formal_words):
                    informal.write(w1)
                    formal.write(w2)
def add_perfix(src_path,tgt_path):
    with open(src_path,'r',encoding='utf-8') as src:
        with open(tgt_path,'w',encoding='utf-8') as tgt:
            for line in src.readlines():
                if len(line.strip('\n').split(' '))<=1:
                    tgt.write(' '+line.strip('\n')+' \n')
                else:
                    tgt.write(line)

if __name__=='__main__':
    src_path='/home/hssun/sunhuashan/PPSeq2Seq/data/source/informal_words.txt'
    tgt_path='/home/hssun/sunhuashan/PPSeq2Seq/data/source/informal_words_2.txt'
    tgt_dir='/home/hssun/sunhuashan/PPSeq2Seq/data/source/'
    split_data(src_path,tgt_dir)
    # add_perfix(src_path,tgt_path)

