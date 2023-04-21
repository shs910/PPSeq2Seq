from train import test,get_args
from data_utils import TextDataset,create_dataloader,split_data
from transformers import BertForSequenceClassification,BertTokenizer

        
def inference():
    args=get_args()
    test_data_path='/home/hssun/sunhuashan/tst4mt/data/zh-en新闻数据/processed/test/apiout_incontext4.news'
    ref_path='/home/hssun/sunhuashan/tst4mt/data/zh-en新闻数据/processed/test/test.news'
    title='apiout_incontext_4_'
    #加载模型、分词器
    model=BertForSequenceClassification.from_pretrained('/home/hssun/sunhuashan/PPSeq2Seq/evaluation/style_classifier/checkpoints',num_labels=2)
    model=model.to(args.device)
    tokenizer=BertTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
    #加载数据
    test_dataset=TextDataset(test_data_path,mode='test',tokenizer=tokenizer,max_length=args.max_length)
    test_dataloader=create_dataloader(test_dataset,mode='test',\
        batch_size=args.batch_size,batchify_fn=test_dataset.collate_func)
    args.tgt_path='/home/hssun/sunhuashan/tst4mt/data/zh-en新闻数据/processed/test/res'+title+'res_with_label.txt'
    tgt_dir='/home/hssun/sunhuashan/tst4mt/data/zh-en新闻数据/processed/test/res'
    test(model,test_dataloader,args)
    
    split_data(args.tgt_path,test_data_path,tgt_dir,title,ref_path)

if __name__=='__main__':
    inference()
    #CUDA_VISIBLE_DEVICES=1 python /home/jcdu/zja/shs/evaluation/style_classifier/inference.py
