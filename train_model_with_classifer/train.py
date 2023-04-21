import torch
import argparse
import os
import time
import json
from train_eval_utils import train_one_epoch,evaluate_performance
from Seq2SeqwithClassifier import Discriminator
from data_utils import TextDataset,get_dataloader,loaddata
from torch.optim import Adam

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 输出路径信息
    if not os.path.exists(args.output_fp):
            os.makedirs(args.output_fp)
    classifier_head_meta_fp = os.path.join(
        args.output_fp, "{}_classifier_head_meta.json".format(args.dataset)
    )# args.dataset表示使用的那个数据集str

    classifier_head_fp_pattern = os.path.join(
        args.output_fp, "{}_classifier_head_epoch".format(args.dataset) + "_{}.pt"
    )

    
    print("Preprocessing {} dataset for training...".format(args.dataset))
    start = time.perf_counter()
    idx2class=['news','oral']
    class2idx = {c: i for i, c in enumerate(idx2class)}

    # 加载模型
    discriminator = Discriminator(
            class_size=len(idx2class),
            pretrained_model=args.pretrained_model,
            model_type=args.paraphrase_model,
            cached_mode=args.cached,
            device=device
        ).to(device)
    print('model done!')
    print('loading..data')
    # 加载数据集
    train_texts,train_labels=loaddata(args.train_path,mode='train')
    train_dataset=TextDataset(texts=train_texts,labels=train_labels,\
        tokenizer=discriminator.tokenizer)
    train_dataloader=get_dataloader(train_dataset,mode='train',batch_size=args.batch_size)

    eval_texts,eval_labels=loaddata(args.eval_path,mode='eval')
    eval_dataset=TextDataset(texts=eval_texts,labels=eval_labels,\
        tokenizer=discriminator.tokenizer)
    eval_dataloader=get_dataloader(eval_dataset,mode='train',batch_size=args.batch_size)
    discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": args.pretrained_model,
            "class_vocab": class2idx,
            "default_class": 1,
        }
    end = time.perf_counter()
    print("Preprocessed {} data points".format(
        len(train_dataset) + len(eval_dataset))
    )
    print("Data preprocessing took: {:.3f}s".format(end - start))

    with open(classifier_head_meta_fp, "w") as meta_file:
        json.dump(discriminator_meta, meta_file)
    
    optimizer = Adam(discriminator.parameters(), lr=args.learning_rate)

    eval_losses = []
    eval_accuracies = []
    best_accuracy=0.0
    steps=[]
    stp=len(train_dataloader)
    print('steps per epoch: {}'.format(stp))
    for epoch in range(args.epochs):
        start = time.perf_counter()
        print("\nEpoch", epoch + 1)

        test_loss_l,test_accuracy_l,step_s=train_one_epoch(
            discriminator=discriminator,
            data_loader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=args.log_interval,
            device=device,
            eval_step=args.valid_interval,
            file_path=classifier_head_fp_pattern,
        )
        eval_losses.extend(test_loss_l)
        eval_accuracies.extend(test_accuracy_l)
        steps.extend(list(map(lambda x:x+epoch*(stp),step_s)))
        end = time.perf_counter()
        print("Epoch took: {:.3f}s".format(end - start))
        # print("\nExample prediction")
        # predict(example_sentence, discriminator, idx2class,
        #         cached=cached, device=device)
    min_loss = float("inf")
    min_loss_epoch = 0
    max_acc = 0.0
    max_acc_epoch = 0
    print("Test performance per epoch")
    print("epoch_step\tloss\tacc")
    for e, (loss, acc) in enumerate(zip(steps,eval_losses, eval_accuracies)):
        print("{}\t{}\t{}".format(e + 1, loss, acc))
        if loss < min_loss:
            min_loss = loss
            min_loss_epoch = e + 1
        if acc > max_acc:
            max_acc = acc
            max_acc_epoch = e + 1
    print("Min loss: {} - Epoch: {}".format(min_loss, min_loss_epoch))
    print("Max acc: {} - Epoch: {}".format(max_acc, max_acc_epoch))


def get_args():
    parser=argparse.ArgumentParser(description='args4train classifier')
    parser.add_argument('--train_path',type=str,default='/home/hssun/sunhuashan/PPSeq2Seq/data/train_data/train_2w.txt')
    parser.add_argument('--eval_path',type=str,default='/home/hssun/sunhuashan/PPSeq2Seq/data/train_data/valid_2w.txt')
    parser.add_argument('--output_fp',type=str,default='/home/hssun/sunhuashan/PPSeq2Seq/results/T5',help='out put dir')
    parser.add_argument('--dataset',type=str,default='oral',help='which dataset used')
    parser.add_argument("--pretrained_model", type=str, default="hetpandya/t5-base-tapaco",
                        help="Pretrained model to use as base")
    parser.add_argument("--paraphrase_model", type=str, default="t5",
                        help="Pretrained model to use as base")                   
    parser.add_argument("--epochs", type=int, default=1, metavar="N",
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learnign rate")
    parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--log_interval", type=int, default=40, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--valid_interval", type=int, default=140, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--cached", type=bool, default=False,
                        help="")
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device_id)
    return args

if __name__=='__main__':
    args=get_args()
    train(args)