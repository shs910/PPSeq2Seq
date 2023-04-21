from data_utils import TextDataset,create_dataloader
from transformers import BertForSequenceClassification,BertTokenizer
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import os
import torch
import argparse

def test(model,data_loader,args):
    model.eval()
    with open(args.tgt_path,'w',encoding='utf-8') as f:
        oral_num=0
        total_num=0
        with torch.no_grad():
            print('testing...')
            for data in tqdm(data_loader):
                encoding=data['encoding']
                input_ids,attention_mask,token_type_ids=encoding['input_ids'].to(args.device),\
                    encoding['attention_mask'].to(args.device),encoding['token_type_ids'].to(args.device)
                prediction=model(input_ids=input_ids,attention_mask=attention_mask,\
                    token_type_ids=token_type_ids)
                logits=prediction.logits
                pres=logits.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
                oral_num+=pres.sum()
                total_num+=len(pres)
                for pre in pres:
                    f.write(str(pre)+'\n')
        print('total_num:{},oral_num:{},ratio:{}'.format(total_num,oral_num,round(oral_num/total_num,4)))

def validation(model, data_loader,args):
    model.eval()
    correct = 0
    total = 0
    accuracy = 0
    loss_t=[]
    with torch.no_grad():
        print('evaluation...')
        for data in tqdm(data_loader):
            labels=data['labels'].to(args.device)
            encoding=data['encoding']
            input_ids,attention_mask,token_type_ids=encoding['input_ids'].to(args.device),\
                    encoding['attention_mask'].to(args.device),encoding['token_type_ids'].to(args.device)
            prediction=model(input_ids=input_ids,attention_mask=attention_mask,\
                token_type_ids=token_type_ids,labels=labels)
            loss=prediction.loss 
            logits=prediction.logits
            pre=logits.data.max(1, keepdim=True)[1]
            correct+=pre.eq(labels.data.view_as(pre)).sum().cpu().numpy()
            total+=len(labels)
            loss_t.append(loss.item())
    accuracy=correct/total
    acc=accuracy.item()
    print('evalustion total num {},correct num {},ACC {}'\
    .format(total,correct,round(acc,4)))
    model.train()
    return round(np.mean(loss_t),4),round(acc,4)


def train(model,data_loader, valid_data_loader, test_data_loader,optimizer,args):
    epoch_num=args.epoch
    train_loss,train_acc,valid_acc,valid_loss=[],[],[],[]
    model.train()#使得torch可以计算梯度等
    best_valid_acc=0
    eval_step=args.eval_step
    optimizer.zero_grad()
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        epoch_loss,epoch_acc=[],[]
        print('train epoch {}'.format(epoch))
        for step, data in enumerate(tqdm(data_loader), 0):
            labels=data['labels'].to(args.device)
            encoding=data['encoding']
            input_ids,attention_mask,token_type_ids=encoding['input_ids'].to(args.device),\
                    encoding['attention_mask'].to(args.device),encoding['token_type_ids'].to(args.device)
            prediction=model(input_ids=input_ids,attention_mask=attention_mask,\
                token_type_ids=token_type_ids,labels=labels)
            loss=prediction.loss 
            logits=prediction.logits  #[bs,nlabels]
            #梯度回传
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pre=logits.data.max(1, keepdim=True)[1]
            correct=pre.eq(labels.data.view_as(pre)).sum().cpu().numpy()
            acc=correct/len(labels)
            epoch_loss.append(loss.item())
            epoch_acc.append(acc.item())
            if step%eval_step==0:
                epoch_mean_loss=round(np.mean(epoch_loss),6)
                epoch_mean_acc=round(np.mean(epoch_acc),6)
                train_loss.append(epoch_mean_loss)
                train_acc.append(epoch_mean_acc)
                epoch_valid_loss,epoch_valid_acc=validation(model,valid_data_loader,args)
                valid_acc.append(epoch_valid_acc)
                valid_loss.append(epoch_valid_loss)
                print('epoch:{},step:{},train loss:{},train acc:{};valid loss:{},valid acc:{}'\
                  .format(epoch,step,epoch_mean_loss,epoch_mean_acc,epoch_valid_loss,epoch_valid_acc))
                if best_valid_acc<epoch_valid_acc:
                #保存模型
                    model.save_pretrained(args.save_dir)
                    print('save best model! valid loss is {}, valids acc is:{}'.format(epoch_valid_loss,epoch_valid_acc))
                    best_valid_acc=epoch_valid_acc
                    test(model,test_data_loader,args)
        
    print('Finished Training!')
    return (train_loss,train_acc,valid_loss,valid_acc)

def get_args():
    parse=argparse.ArgumentParser(description='train args')
    parse.add_argument('--data_dir',type=str,default='/home/jcdu/zja/shs/data_creation/data/classifer_data',help='train\eval\\test data path')
    parse.add_argument('--epoch',type=int,default=1)
    parse.add_argument('--batch_size',type=int,default=64*3)
    parse.add_argument('--num_labels',type=int,default=2)
    parse.add_argument('--label_smothing',type=float,default=0.1)
    parse.add_argument('--lr',type=float,default=5e-5)
    parse.add_argument('--weight_decay',type=float,default=0.001)
    parse.add_argument('--device',type=str,default='cuda:0')
    parse.add_argument('--max_length',type=int,default=128)
    parse.add_argument('--eval_step',type=int,default=500)
    parse.add_argument('--tgt_path',type=str,default='/home/jcdu/zja/shs/evaluation/style_classifier/results/oral_pre.en')
    parse.add_argument('--save_dir',type=str,default='/home/jcdu/zja/shs/evaluation/style_classifier/checkpoints')
    args=parse.parse_args()

    return args

def main():
    args=get_args()
    #加载模型、分词器
    model=BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity",num_labels=args.num_labels)
    model=model.to(args.device)
    tokenizer=BertTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
    #加载数据
    train_path=os.path.join(args.data_dir,'train.txt')#记得修改!!!
    valid_path=os.path.join(args.data_dir,'valid.txt')
    test_path=os.path.join(args.data_dir,'test.txt')
    train_dataset=TextDataset(train_path,mode='train',tokenizer=tokenizer,max_length=args.max_length)
    valid_dataset=TextDataset(valid_path,mode='valid',tokenizer=tokenizer,max_length=args.max_length)
    test_dataset=TextDataset(test_path,mode='test',tokenizer=tokenizer,max_length=args.max_length)
    train_dataloader=create_dataloader(train_dataset,mode='train',\
        batch_size=args.batch_size,batchify_fn=train_dataset.collate_func)
    valid_dataloader=create_dataloader(valid_dataset,mode='valid',\
        batch_size=args.batch_size,batchify_fn=valid_dataset.collate_func)
    test_dataloader=create_dataloader(test_dataset,mode='test',\
        batch_size=args.batch_size,batchify_fn=test_dataset.collate_func)

    #加载损失函数和优化器
    #criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smothing)
    optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),\
    lr=args.lr,betas=(0.9,0.99),weight_decay=args.weight_decay)
    train_res=train(model,train_dataloader,valid_dataloader,test_data_loader=test_dataloader,optimizer=optimizer,args=args)

if __name__=='__main__':
    main()
