import torch
import os
import time
import copy
import numpy as np
import torch.nn.functional as F
from tqdm import trange
from operator import add
from torch.autograd import Variable
from classifier import ClassifierHead
from typing import List, Optional, Tuple, Union

SMALL_CONST = 1e-15
BIG_CONST = 1e10

def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    # 将top-k的位置外设置为inf，使得exp(-inf)~0,对分母无贡献（softmax）
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        # 将小于的预测mask掉（设置为0/-infty）
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)

def get_classifier(class_size,embed_size,device: str,classifier_model_path):

    classifier = ClassifierHead(
        class_size=class_size,
        embed_size=embed_size
    ).to(device)
    # 加载训练好的模型
    classifier.load_state_dict(
        torch.load(classifier_model_path, map_location=device))
    classifier.eval()
    print('classifier loaded!!')

    return classifier
    
def tuple2list4past(past,seq_len):
    # shape tuple(tupe(4*tensor))
    '''
    保留decoder的KV
    '''
    decoder_past=[]
    encoder_past=[]
    n_layers=len(past)
    # print('n_layers',n_layers)
    for i in range(n_layers):
        decoder_kv=[]
        encoder_kv=[]
        for j in range(4):
            # print(past[i][j].shape[2])
            if past[i][j].shape[2]!=seq_len or (past[i][j].shape[2]==seq_len and j <2):
                decoder_kv.append(past[i][j])
            else:
                encoder_kv.append(past[i][j])
        encoder_past.append(torch.stack(encoder_kv))
        decoder_past.append(torch.stack(decoder_kv))

    return encoder_past,decoder_past
def list2tuple4past(encoder_past,decoder_past):
    ''''''
    n_layers=len(encoder_past)
    # print('n_layers',n_layers)
    past=[]
    for i in range(n_layers):
        kv_item=tuple(x for x in decoder_past[i])+tuple(x for x in encoder_past[i])
        past.append(kv_item)

    past=tuple(past)
    # print('====back===')
    # print(len(past))
    # print(len(past[0]))
    return past

def perturb_past(   past,
                    model,
                    last,
                    t_step,
                    encoder_outputs,
                    unpert_past,
                    unpert_logits,
                    accumulated_hidden,
                    grad_norms,
                    stepsize,
                    classifier,
                    unfinished_sequences,
                    one_hot_bows_vectors,
                    args,
                    token_loss=0,
                ):
    # print('====before===')
    # print(len(past))
    # print(len(past[0]))
    # if t_step<2:
    #     print('='*7,'past_k_v','='*7)
    #     print('len')
    seq_len=encoder_outputs[0].shape[1]
    #预处理past_KV 为[n_lyasers*tensor(4,batch_size, num_heads, sequence_length, embed_size_per_head)]
    encoder_past,past=tuple2list4past(past,seq_len=seq_len)
    # 初始化 \Delta H_t
    
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]
    if accumulated_hidden is None:
        #最后一层hidden_state(迭代时需要累加修改)
        accumulated_hidden = 0
    
    if args.decay:
        decay_mask = torch.arange(0.,1.0 + SMALL_CONST,1.0 / (args.window_length))[1:]
    else:
        decay_mask = 1.0
    # print(past[0].shape)
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > args.window_length and args.window_length > 0:
        # 仅对窗口内的H进行优化（会滑动？）
        ones_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([args.window_length])
                + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([curr_length - args.window_length])
                + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(args.device)
    else:
        # 全部优化
        window_mask = torch.ones_like(past[0]).to(args.device)
    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    # 迭代计算 \Delta H_t
    perturb_token_loss=0.0
    count=0
    for i in range(args.num_iterations):
        count+=1
        # print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=args.device)
            for p_ in grad_accumulator
        ]
        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))

        perturbed_past=list2tuple4past(encoder_past,perturbed_past)

        _, _, _, curr_length, _ = curr_perturbation[0].shape
        # 计算修改后LM的输出（）(logit,past_KV,hidden_state)past_KV忽略
        # print(last.max())
        # print(la)
        outputs= model(encoder_outputs=encoder_outputs,\
                decoder_input_ids = last, past_key_values=perturbed_past,return_dict=True)
        #print(outputs.keys())
        all_logits, all_hidden =outputs.logits,outputs.decoder_hidden_states
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        #修改后的预测
        logits = all_logits[:, -1, :]  # 最后一个：当前预测的token
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []
    

        # 计算两个分布的KL损失
        kl_loss = 0.0
        if args.kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(args.device).detach()
            )
            # 修正了
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(args.device).detach()
            corrected_probs = probs + correction.detach()
            # print('corrected_probs:' ,corrected_probs.shape)
            # print('corrected_probs[0]:' ,corrected_probs[0].unsqueeze(0).shape)
            # print('unpert_probs:',unpert_probs.shape)
            # print('unpert_probs[0]:',unpert_probs[0].unsqueeze(0).shape)
            # kl_loss = torch.sum(args.kl_scale * (
            #     (corrected_probs * (corrected_probs / unpert_probs).log())
            # ),dim=1)
            # if verbosity_level >= VERY_VERBOSE:
            #     print(' kl_loss', kl_loss.data.cpu().numpy())
            # print('kl_loss: ',kl_loss )
            
            # print('loss: ',loss)
            

        
        # if verbosity_level >= VERBOSE:
        #     print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        # 每个sample的loss分别计算并回传
        # bath的loss不进行平均
        # 需要计算分类器损失时
        if not args.bag_of_words or args.loss_type!=2:
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)

            # 获得模型embedding层
            if args.model_type=='pegasus':
                wte=model.model.resize_token_embeddings()
            elif args.model_type=='t5':
                wte = model.resize_token_embeddings()
            #  多次求平均：？？？
            for _ in range(args.horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                
                outputs= model(
                    encoder_outputs=encoder_outputs,
                    past_key_values=curr_unpert_past,
                    decoder_inputs_embeds=inputs_embeds,
                    return_dict=True
                )
                curr_unpert_past, curr_all_hidden=outputs.past_key_values,outputs.decoder_hidden_states
                curr_hidden = curr_all_hidden[-1]
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                    curr_hidden, dim=1)
            # t1=time.perf_counter()
            prediction = classifier(new_accumulated_hidden /
                                        (curr_length + 1 + args.horizon_length))
            batch_size=prediction.shape[0]
            label = torch.tensor( batch_size* [args.class_label],
                                    device=args.device,
                                    dtype=torch.long)
            # print('prediction ',prediction.shape)
            # print('prediction[0] ',prediction[0].unsqueeze(0).shape)
            # print('label ',label.shape)
            # print('label[0] ',label[0].unsqueeze(0).shape)
            # print('discrim_loss:',discrim_loss)
            #print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
        num_layers=len(grad_accumulator)
        # =======bath 单独计算修改代码部分=========
        # grad_norms=[]
        if not args.loss_not_divded :
            for batch_idx in range(batch_size):
                # 如果已经停止生成，则不再计算该部分
                if unfinished_sequences[batch_idx] == 0:
                    continue
                # 计算损失
                indx_window_mask=window_mask[:,batch_idx,:,:,:].unsqueeze(dim=1)
                # 计算损失
                ce_loss = torch.nn.CrossEntropyLoss()
                discrim_loss = ce_loss(prediction[batch_idx].unsqueeze(0), label[batch_idx].unsqueeze(0))
                kl_loss = args.kl_scale * (
                    (corrected_probs[batch_idx].unsqueeze(0) * \
                        (corrected_probs[batch_idx].unsqueeze(0) / unpert_probs[batch_idx].unsqueeze(0)).log()).sum()
                )
                loss=discrim_loss+kl_loss
                
                if batch_idx==batch_size-1:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                idx_curr_perturbation_grad=[layer_curr_perturbation.grad[:,batch_idx,:,:,:].unsqueeze(dim=1)\
                    for layer_curr_perturbation in curr_perturbation]
                # print('idx_curr_perturbation_grad[0]: ',idx_curr_perturbation_grad[0].shape)
                idx_grad_accumulator=[np.expand_dims(layer_grad_accumulator[:,batch_idx,:,:,:],axis=1) \
                    for layer_grad_accumulator in grad_accumulator]
                # print('idx_grad_accumulator[0]: ',idx_grad_accumulator[0].shape)
                # 计算norm和梯度等：
                idx_grad_norms = [
                    (torch.norm(grad * indx_window_mask) + SMALL_CONST)
                    for index, grad in enumerate(idx_curr_perturbation_grad)
                    ]
                idx_grad = [
                        -stepsize *
                        (grad * indx_window_mask / idx_grad_norms[
                            index] ** args.gamma).data.cpu().numpy()
                        for index, grad in enumerate(idx_curr_perturbation_grad)
                    ]
                # print('idx_grad_accumulator[0]:',idx_grad_accumulator[0].shape)
                # print('grad:',idx_grad[0].max())
                idx_grad_accumulator = list(map(add, idx_grad, idx_grad_accumulator))
                # print('idx_grad_accumulator[0] after shape',idx_grad_accumulator[0].shape)
                # 更新原来的grad_accumulator
                for layer_idx in range(num_layers):
                    grad_accumulator[layer_idx][:,batch_idx,:,:,:]=np.squeeze(idx_grad_accumulator[layer_idx],axis=1)
                # 梯度在最后清零
            # =======bath 单独计算修改代码部分=========
        else:
            # batch-mean的方法计算loss

        # 计算梯度，并更新\Delta H_t += \alpha * gradient
        # print('len_curr_perturbation: ',len(curr_perturbation))
        # print('curr_perturbation[0]: ',curr_perturbation[0].shape)
        # 源码
            if args.bag_of_words and args.loss_type!=1:
                # print('Bow损失')
                for one_hot_bow in one_hot_bows_vectors:
                    # print('probs;size {}'.format(probs.shape))
                    # print('one_hot_bow;size {}'.format(one_hot_bow.shape))
                    # exit()
                    bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                    bow_loss = -torch.log(torch.sum(bow_logits))
                    loss += bow_loss
                    loss_list.append(bow_loss)
            if args.loss_type!=2:
                # 计算损失
                # print('交叉上损失')
                ce_loss = torch.nn.CrossEntropyLoss()
                discrim_loss = ce_loss(prediction, label)
                loss += discrim_loss
                loss_list.append(discrim_loss)

            kl_loss = args.kl_scale * (
                    (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
                )
            loss += kl_loss
            loss_per_iter.append(loss.mean().data.cpu().numpy())
            loss.backward()
            # print('loss:',loss)
            # 计算grad_norms：
            if args.loss_type==2 and grad_norms is not None:
                # print('grad_norms is not None')
                grad_norms = [
                        torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                        for index, p_ in enumerate(curr_perturbation)
                    ]
            else:
                # print('grad_norms')
                grad_norms = [
                    (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                    for index, p_ in enumerate(curr_perturbation)
                ]
            # print('grad_norms:',grad_norms[0])
        # normalize gradients
        # \alpha * gradient
        # 源码
            grad = [
                -stepsize *
                (p_.grad * window_mask / grad_norms[
                    index] ** args.gamma).data.cpu().numpy()
                for index, p_ in enumerate(curr_perturbation)
            ]
            # if args.loss_type==2 :
            #     grad=list(map(lambda x:x*7,grad))
        # print('grad:',grad[0].shape)
        # accumulate gradient
        # \Delta H_t += [\alpha * gradient]
        # 源码
            grad_accumulator = list(map(add, grad, grad_accumulator))
        # print('len grad_accumulator:',len(grad_accumulator))
        # print('grad_accumulator[0].sise',grad_accumulator[0].shape)
        # exit()
        # reset gradients, just to make sure
        # 梯度清零
        for p_ in curr_perturbation:
            p_.grad.data.zero_()
        # ////一次梯度计算+清0零
        # removing past from the graph(detach,迭代重新使用)
        # 值未变
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past
        # t2=time.perf_counter()
        # print('loss and back using {}'.format(t2-t1))
        #想查看loss
        perturb_token_loss+=loss.mean()

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=args.device)
        for p_ in grad_accumulator
    ]
    # H_t+=\Delta H_t
    pert_past = list(map(add, past, grad_accumulator))

    pert_past=list2tuple4past(encoder_past,pert_past)
    # print('loss:',perturb_token_loss.item()/count)
    token_loss+=perturb_token_loss.item()/count
    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter,token_loss


def generate_text_pplm(model,tokenizer,classifier,input_text,args,perturb=False,bow_indices=None):
    '''
    input_text={}:分词后的模型输入
    '''
    if not args.not_log:
        print('perturb:',perturb)
    #分词：
    # input_text为batch data
    # max_length=args.max_length
    # collect one hot vectors for bags of words
    one_hot_bows_vectors=None
    if args.loss_type!=1 and args.bag_of_words and perturb:
        one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer,args,
                                                      args.device)
    input_text=tokenizer(input_text,padding=True,truncation=True,max_length=args.max_length,return_tensors='pt').to(args.device)
    batch_size,input_seq_len=input_text.input_ids.shape[0],input_text.input_ids.shape[1]
    output_so_far = None  # 记录模型的输出id
    decoder_start_token_id=model.config.decoder_start_token_id
    output_so_far=[[decoder_start_token_id] for _ in range(batch_size)]
    output_so_far=torch.tensor(output_so_far,dtype=torch.int64).to(args.device)

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []
    if not args.not_log:
        range_func = trange(args.length, ascii=True)
    else:
        range_func = range(args.length)

    # endocer获得encoder-hidden_states
    if args.model_type=='pegasus':
        encoder=model.get_encoder()
        encoder_outputs = encoder(
                input_ids=input_text.input_ids,
                attention_mask=input_text.attention_mask,
                return_dict=True,output_hidden_states=True
            )
        # print(encoder_outputs.keys())
    elif args.model_type=='t5':
        encoder_outputs = model.encoder(
                input_ids=input_text.input_ids,
                attention_mask=input_text.attention_mask,
                return_dict=True,output_hidden_states=True
            )
    #     print(type(encoder_outputs))
    #     print(encoder_outputs.keys())
    # ['last_hidden_state', 'hidden_states']
    # 用于检测那些句子还没有生成完毕
    unfinished_sequences = output_so_far.new(output_so_far.shape[0]).fill_(1)
    # past 初始化None
    # 进行decoder
    past=None
    if not args.not_log:
        print('begin generation')
    tokens_loss_mean=[]
    for i in range_func:
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            # 不存在past_key_values时
            decoder_outputs= model(encoder_outputs=encoder_outputs,\
                decoder_input_ids=last,return_dict=True)
            past=decoder_outputs.past_key_values
            # 该版本的transformers返回：
            # logits、past_key_values、hidden_states
            #(batch_size, sequence_length, config.vocab_size)
            # [(2, batch_size, num_heads, sequence_length, embed_size_per_head) * n_layers]
            # (batch_size, sequence_length, hidden_size)
        
        # 普通的生成
        outputs = model(encoder_outputs=encoder_outputs,\
                decoder_input_ids=output_so_far,return_dict=True)
        #print(outputs.keys())
        unpert_logits, unpert_past, unpert_all_hidden=outputs.logits,outputs.past_key_values\
            ,outputs.decoder_hidden_states
        # print(type(unpert_all_hidden))
        # print(len(unpert_all_hidden))
        unpert_last_hidden = unpert_all_hidden[-1]
        # if i < 2:
        #     print('last_hidden:size',unpert_last_hidden.shape)


        # check if we are abowe grad max length
        if i >= args.grad_length:
            current_stepsize = args.stepsize * 0
        else:
            current_stepsize = args.stepsize

        # modify the past if necessary
        if not perturb or args.num_iterations == 0:
            # 不修改内部状态🤐
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)
            # if i<2:
            #     print('accumulated_hidden size ',accumulated_hidden.shape)
            if past is not None and i>=args.strat_perturb:
                # 最初的past is None
                pert_past, _, grad_norms, loss_this_iter,token_loss = perturb_past(   
                    past,
                    model,
                    last,
                    t_step=i,
                    encoder_outputs=encoder_outputs,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    classifier=classifier,
                    unfinished_sequences=unfinished_sequences,
                    one_hot_bows_vectors=one_hot_bows_vectors,
                    args=args,
                    token_loss=0.0
                )
                tokens_loss_mean.append(token_loss)
                loss_in_time.append(loss_this_iter)
                # print('loss_this_iter ',np.mean(loss_this_iter))
            else:
                pert_past = past
        
        # modify past的KV后再输入预测
        # last 为之前的token相关参数，past为修改后的K-V
        out= (decoder_input_ids=last, \
            past_key_values=pert_past,encoder_outputs=encoder_outputs,\
                return_dict=True)
        # print(out.keys())
        # perturb后的past代替之前的past
        pert_logits, past, pert_all_hidden=out.logits,out.past_key_values,out.decoder_hidden_states
        # temperature：缩放
        pert_logits = pert_logits[:, -1, :] / args.temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

        if classifier is not None:
            #计算 p(a|x)的损失
            ce_loss = torch.nn.CrossEntropyLoss()
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([args.class_label for _ in range(batch_size)], device=args.device,
                                 dtype=torch.long)
            unpert_discrim_loss = ce_loss(prediction, label)
            if not args.not_log:
                print("unperturbed discrim loss",unpert_discrim_loss.data.cpu().numpy())
        else:
            unpert_discrim_loss = 0
        
        # Fuse the modified model and original model
        if perturb:
            # 修改KV后，Post-norm Geometric Mean Fusion！！！！
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = ((pert_probs ** args.gm_scale) * (
                    unpert_probs ** (1 - args.gm_scale)))  # + SMALL_CONST
            
            pert_probs = top_k_filter(pert_probs, k=args.top_k,
                                      probs=True)  # + SMALL_CONST

            # rescale！！！
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=args.top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if not args.not_sample:
            last = torch.multinomial(pert_probs, num_samples=1)
        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)
        #先铺平：
        last=last.reshape(-1)
        #追踪已经生成完毕的数据
        last = last * unfinished_sequences + args.pad_token_id * (1 - unfinished_sequences)
        # if eos_token was found in one sentence, set sentence to finished
        unfinished_sequences = unfinished_sequences.mul((last != args.eos_token_id).long())
        last=last[:,None]
        
        
        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )
        if not args.not_log:
            print(tokenizer.decode(output_so_far.tolist()[0]))
        # 判断是否所有句子均生成完毕，退出生成：
        if unfinished_sequences.max() == 0:
            break
    
    return output_so_far, unpert_discrim_loss, loss_in_time,np.mean(tokens_loss_mean)

def get_paraphrases(sentence,tokenizer,model, prefix="paraphrase: ", n_predictions=5, top_k=120, max_length=256,device="cuda"):
        text = prefix + sentence + " </s>"
        encoding = tokenizer.encode_plus(
            text, pad_to_max_length=True, return_tensors="pt"
        )
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding[
            "attention_mask"
        ].to(device)

        model_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            do_sample=True,
            max_length=max_length,
            top_k=top_k,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=n_predictions,
        )

        outputs = []
        for output in model_output:
            generated_sent = tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            if (
                generated_sent.lower() != sentence.lower()
                and generated_sent not in outputs
            ):
                outputs.append(generated_sent)
        return outputs

def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str], tokenizer) -> \
        List[List[List[int]]]:
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        filepath = id_or_path
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        bow_indices.append(
            [tokenizer.encode(word.strip(),
                              add_special_tokens=False)
             for word in words])
    # print('===========bow_indices=============')
    # print('[n_file*[n_line*[n_line_words]]]')
    # print(len(bow_indices))
    # for id in bow_indices[0]:
    #     print(tokenizer.decode(id))
    # print('='*20)
    return bow_indices

def build_bows_one_hot_vectors(bow_indices, tokenizer, args,device='cuda'):
    if bow_indices is None:
        return None
    # print('='*7,'one_hot_bows_vectors','='*7)
    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        # 只保留了单个words的词汇
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        print(num_words,tokenizer.vocab_size)
        one_hot_bow = torch.zeros(num_words, args.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)

    return one_hot_bows_vectors

def full_text_generation(model,tokenizer,classifier,input_text,bag_of_words,args):
    # 生成为原始文本
    unpert_gen_tok_text=['nothing']
    # BoW
    bow_indices = []
    if bag_of_words and args.loss_type!=1:
        print(bag_of_words.split(";"))
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),tokenizer)
    unpert_loss=0
    if not args.not_gen_both:
        unpert_gen_tok_text, _, _ ,unpert_loss= generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text,
            classifier=classifier,
            args=args,
            perturb=False
        )
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    
    # 生成修改后的文本
    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []
    for i in range(args.num_samples):
        # past==None
        pert_gen_tok_text, discrim_loss, loss_in_time ,pert_loss= generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text,
            classifier=classifier,
            args=args,
            bow_indices=bow_indices,
            perturb=True,
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if args.device == 'cuda':
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time,unpert_loss,pert_loss

def load_data(src_path,bath_size,model_type):
    print('laoding data....')
    with open(src_path,'r',encoding='utf-8') as f:
        data=[]
        bath_data=[]
        bath_count=0
        total_leng=0
        for idx,line in enumerate(f.readlines()):
            line=line.strip('\n')
            total_leng=idx+1
            if model_type=='t5':
                bath_data.append("paraphrase: "+line+" </s>")
            elif model_type=='pegasus':
                bath_data.append(line)
            bath_count+=1
            if bath_count==bath_size:
                data.append(copy.deepcopy(bath_data))
                bath_count=0
                # print(bath_data)
                bath_data.clear()
        if bath_data:
            data.append(bath_data)
        print('total_num:{},bath_szie:{},num_bath:{}'.format(\
            total_leng,bath_size,len(data)))
        # print(data)
        # exit()
    return data

def save_results(unpert_data,pert_data,tgt_dir,title):
    unpert_path=os.path.join(tgt_dir,title+'withoutClassifier.txt')
    pert_path=os.path.join(tgt_dir,title+'withClassifier.txt')
    if len(unpert_data)!=len(pert_data):
        unpert_data=[['nothing'] for _ in range(len(pert_data))]
    with open(unpert_path,'a',encoding='utf-8') as unpert,\
        open(pert_path,'a',encoding='utf-8') as pert:
        #print(unpert_data,pert_data)
        for un_text,texts in zip(unpert_data,pert_data):
            for text in un_text:
                unpert.write(text+'\n')
            # batch infer 时，unpert_data也是bath！！！
            for text in texts:
                pert.write(text+'\n')