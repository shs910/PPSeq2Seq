import argparse
import torch
import numpy as np
import time
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer,T5ForConditionalGeneration, T5Tokenizer
from inference_utils import full_text_generation,get_paraphrases,load_data,save_results,get_classifier

def inference(args):
    # set Random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set the device
    device = "cuda" if torch.cuda.is_available()  else "cpu"
    
    # load pretrained model for paraphrase
    if  args.model_type=='t5' and (args.model_type in args.pretrained_model):
        model = T5ForConditionalGeneration.from_pretrained(
            args.pretrained_model,
            output_hidden_states=True
        )
        # load tokenizer
        tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model)
        print(args.pretrained_model)
        print('T5 base')
        print('vocab_size: ',model.config.vocab_size)

    elif args.model_type=='pegasus' and (args.model_type in args.pretrained_model):
        model = PegasusForConditionalGeneration.from_pretrained(
            args.pretrained_model,
            output_hidden_states=True
        )
        # load tokenizer
        tokenizer = PegasusTokenizer.from_pretrained(args.pretrained_model)
        args.length=model.config.max_length
        print('max_inputs_length:',tokenizer.model_max_length)
        print('gen_max_length: ',args.length)
        print('pegasus base')
    else:
        raise Warning('model type error')
    model.to(device)
    model.eval()
    if model.config.vocab_size!=tokenizer.vocab_size:
        print('vocab_size is unequal')
        print('model.config.vocab_size: ',model.config.vocab_size)
        print('tokenizer.vocab_size: ',tokenizer.vocab_size)
    args.vocab_size=model.config.vocab_size
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False
    # 分类器
    classifier = get_classifier(args.class_size,model.config.hidden_size,args.device,args.classifier_model_path)
    args.max_length=tokenizer.model_max_length
    args.eos_token_id=tokenizer.eos_token_id
    args.pad_token_id=tokenizer.pad_token_id
    print('max_input_length:',args.max_length)
    print('eos_token_id: ',args.eos_token_id)
    print('pad_token_id: ',args.pad_token_id)
    print('begin......')
    if args.not_batch_inference:
        texts=["paraphrase: "+args.input_text+" </s>"]
    else:
        texts=load_data(args.inference_path,args.batch_size,args.model_type)
        # 变换为list[batch_data[str]]
    for text in tqdm(texts):
        unpert_texts=[]
        pert_texts=[]
            # start_time=time.perf_counter()
        unpert_gen_tok_text, pert_gen_tok_texts, _, _ ,unpert_loss,pert_loss= full_text_generation(
                model=model,
                tokenizer=tokenizer,
                classifier=classifier,
                input_text=text,
                bag_of_words=args.bag_of_words,
                args=args
            )
            # untokenize unperturbed text
        unpert_gen_text=[]
        if  not args.not_gen_both:
            unpert_gen = tokenizer.batch_decode(unpert_gen_tok_text,skip_special_tokens=True)
            unpert_gen_text.extend(unpert_gen)
            unpert_texts.append(unpert_gen_text)
            if not args.not_log:
                print("= Unperturbed generated text =")
                print(unpert_gen_text)
                print()
                print("= Unperturbed generated text =(using generate)")
            print('unpert_loss:',unpert_loss)
        if args.api_or_not:
            print('==========using generate funciton======')
            paraphrases=get_paraphrases(sentence=args.input_text,tokenizer=tokenizer,\
                    model=model)
            if not args.not_log:
                for sent in paraphrases:
                    print(sent)
                print()
        generated_texts = []
            # iterate through the perturbed texts
            # print('pert_gen_tok_texts shape: ',pert_gen_tok_texts.shape)
            # pert_gen_tok_texts为n_pre的list[tensor] tensor:[batchsize,seq_gen]
        for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
                # try:
                #     print(pert_gen_tok_text.shape)
                # except:
                #     pass
            pert_gen_text = tokenizer.batch_decode(pert_gen_tok_text,skip_special_tokens=True)
            generated_texts.extend(pert_gen_text)
            if not args.not_log:
                print("= Perturbed generated text {} =".format(i + 1))
                print(pert_gen_text)
                print()
        # print('pert_loss:',pert_loss)
        pert_texts.append(generated_texts)
            # end_time=time.perf_counter()
            # print('time used for a batch: {}/s'.format(round(end_time-start_time,4)))
            #print(unpert_texts)
            # 
            # print(len(unpert_texts))
            # print(len(unpert_texts[0]))
            # print(len(pert_texts))
            # print(len(pert_texts[0]))
        if not args.not_save_res:
            save_results(unpert_texts,pert_texts,args.tgt_path,args.file_title)


def get_args():
    parser = argparse.ArgumentParser()
    # 使用的复述模型
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="hetpandya/t5-base-tapaco",
        help="pretrained model name or path to local checkpoint",
    )
    # tuner007/pegasus_paraphrase
    # hetpandya/t5-base-tapaco
    # ramsrigouthamg/t5-large-paraphraser-diverse-high-quality
    parser.add_argument('--model_type',type=str,default='t5')
    # 训练好的分类器地址
    parser.add_argument(
        "--classifier_model_path",
        type=str,
        default="/home/hssun/sunhuashan/PPSeq2Seq/results/T5/oral_classifier_head_epoch_8_best.pt",#epoch_2_
        help="classifier model's path to local checkpoint",
    )
    # BOW模型，目前没用
    parser.add_argument("--bag_of_words", type=str, default='/home/hssun/sunhuashan/PPSeq2Seq/data/source/filler_words_2.txt;/home/hssun/sunhuashan/PPSeq2Seq/data/source/informal_words.txt',help='path to wordlist')
    # 复述几条样本？
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    # 风格
    parser.add_argument(
        "--oral",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "oral", "toxicity", "generic"),
        help="Discriminator to use",
    )
    # Delta H的计算时的迭代次数
    parser.add_argument("--num_iterations", type=int, default=5)   #10 for cla

    parser.add_argument("--grad_length", type=int, default=10000)

    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")

    # important parameters
    parser.add_argument("--gamma", type=float, default=1.5) #1.5for BoW  #1:for classifier,
    parser.add_argument("--gm_scale", type=float, default=0.9) #0.9   #0.99 for Bow #0.95 cla
    parser.add_argument("--kl_scale", type=float, default=0.01) #0.01
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=70)
    parser.add_argument("--stepsize", type=float, default=0.02) # 0.02 #0.03 bow #0.04 classifier
    # Dleta H更新的大小
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10) # 10

    parser.add_argument("--class_size", type=int, default=2)
    # parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--device", type=str, default='cuda')

    parser.add_argument("--api_or_not", action='store_true',help='是否用API输出')
    parser.add_argument("--not_log", action='store_true',help='是否输出过程')
    # loss_type:{1:classifier_noly,2:bow_only,3:both}
    parser.add_argument("--loss_type", type=int,default=1,choices=[1,2,3],help='是否输出过程')
    parser.add_argument("--not_sample", action='store_true',help='sample or greedy')
    parser.add_argument("--class_label", type=int, default=1,help='sample or greedy')

    parser.add_argument("--device_id", type=int, default=4)
    # 不产生复述结果
    parser.add_argument("--not_gen_both", action='store_true',help='同时为用分类器的数据')
    # 以单条样本进行输出
    parser.add_argument("--not_batch_inference", action='store_true',help='')
    parser.add_argument("--batch_size", type=int, default=64,help='')
    # 一个batch的loss进行平均（效果不好）
    parser.add_argument('--loss_not_divded',action='store_true',help=' batch中数据是否单独计算loss')
    # 推理使用的测试集
    parser.add_argument("--inference_path", type=str, default='/home/hssun/sunhuashan/tst4mt/data/zh-en新闻数据/processed/test/news.en',help='path for inference')
    # 结果写入的路径
    parser.add_argument("--tgt_path", type=str, default='/home/hssun/sunhuashan/tst4mt/data/zh-en新闻数据/processed/stylisted',help='path to store results')
    
    # 一些例子
    TEST_INPUTS={1:'The Committee agrees with this proposal.',\
        2:'The Chairman made a statement after which the Committee concluded its work.',\
        3:'However, the scientific quality of the data is variable and not always known.',\
        4:'They can be contacted at the following address',\
        5:'An initial order for 14 Ariane 5 satellites has been signed.',\
        6:'These principles relate to four main areas, discussed below.'}
    parser.add_argument("--input_text", type=str, \
            default=TEST_INPUTS[3])
    # 开始进行隐空间修改的步数
    parser.add_argument('--strat_perturb',type=int,default=0,help='when to perturb the latent space in generation')
    parser.add_argument("--file_title", type=str, default='_test_oral_')
    # 不保存结果
    parser.add_argument('--not_save_res',action='store_true')
    
    # The Committee agrees with this proposal.
    # The Chairman made a statement after which the Committee concluded its work.
    # However, the scientific quality of the data is variable and not always known.
    args=parser.parse_args()
    print('log out: ',not args.not_log)
    print('gen both :', (not args.not_gen_both))
    print('batch_inference: ',(not args.not_batch_inference))
    print('save results:',(not args.not_save_res))
    print('loss use mean :',args.loss_not_divded)
    torch.cuda.set_device(args.device_id)
    print('cuda_device_id:',args.device_id)
    return args

if __name__=='__main__':
    args=get_args()
    inference(args)
    # CUDA_VISIBLE_DEVICES=2
    # pyhton /home/hssun/sunhuashan/PPSeq2Seq/inference/infer.py --not_log --loss_not_divded --file_title t5_1000_
    
    # pyhton /home/hssun/sunhuashan/PPSeq2Seq/inference/infer.py \
    #  --pretrained_model tuner007/pegasus_paraphrase \
    # --model_type pegasus \
    # --classifier_model_path /home/hssun/sunhuashan/PPSeq2Seq/results/Pegasus/oral_classifier_head_best.pt \
    #  --not_log  --file_title pegesus_1000_