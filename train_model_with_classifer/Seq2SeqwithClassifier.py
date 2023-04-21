'''
Seq2SeqModel+Classifier
'''
import torch
import torch.nn  as nn
import math
import torch.nn.functional as F
from classifier import ClassifierHead
from transformers import PegasusForConditionalGeneration, \
    PegasusTokenizer,T5ForConditionalGeneration, T5Tokenizer,\
        AutoTokenizer,AutoModelForSeq2SeqLM

EPSILON = 1e-10
PEGASUS_PATH="tuner007/pegasus_paraphrase"
T5_PATH="hetpandya/t5-base-tapaco"
pre_train_path={'t5':T5_PATH,'pegasus':PEGASUS_PATH}

class Discriminator(nn.Module):
    '''
    pagesus(decoder+classifier)
    '''
    def __init__(self,
            pretrained_model='',
            model_type='pegasus',
            class_size=None,
            classifier_head=None,
            cached_mode=False,
            device='cuda') -> None:
        super(Discriminator,self).__init__()
        #复述模型
        print(pretrained_model)
        if pretrained_model=='':
            pretrained_model=pre_train_path.get(model_type,PEGASUS_PATH)
            print('paraphrase model: ',pretrained_model)
        else:
            assert model_type in pretrained_model
        self.model_type=model_type
        if model_type=='pegasus':
            self.tokenizer=PegasusTokenizer.from_pretrained(pretrained_model)
            self.paraphraser=PegasusForConditionalGeneration.from_pretrained(pretrained_model)
        elif model_type=='t5':
            # try:
            
            self.paraphraser=T5ForConditionalGeneration.from_pretrained(pretrained_model)
            self.tokenizer=T5Tokenizer.from_pretrained(pretrained_model)
            # except:
            print(pretrained_model)
            #     self.tokenizer=AutoTokenizer.from_pretrained(pretrained_model)
            #     self.paraphraser=AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
        else:
            raise Warning('model type: {} is not supportted'.format(self.model_type))
        print('return dict: ',self.paraphraser.config.return_dict)
        print('model_max_length',self.tokenizer.model_max_length)
        print('generate_max_length: ',self.paraphraser.config.max_length)
        #print('decoder num layers :',self.paraphraser.config.decoder_layers)
        self.embed_size=self.paraphraser.config.hidden_size
        self.pad_token_id=self.tokenizer.pad_token_id # 0
        #分类器
        if classifier_head is not None:
            self.classifier=classifier_head
        else:
            self.classifier=ClassifierHead(class_size=class_size,embed_size=self.embed_size)
        self.device = device
        self.cached_mode=cached_mode
        #self.paraphraser.generate
    
    def get_classifier(self):
        '''返回分类器'''
        return self.classifier
    
    def train_custom(self):
        '''冻结LM的参数，仅训练分类器！！！！'''
        for param in self.paraphraser.parameters():
            param.requires_grad = False
        self.classifier.train()

    def create_T5_decoder_bos(self,batch_size):
        decoder_inputs=[[self.pad_token_id] for _ in range(batch_size)]
        decoder_inputs=torch.tensor(decoder_inputs,dtype=torch.int64)
        return decoder_inputs.to(self.device)

    def avg_representation(self,inputs):
        '''
        获得最后一层的平均表示
        '''
        # create mask
        x=inputs["input_ids"]
        attention_mask=inputs['attention_mask']
        mask=x.ne(self.pad_token_id).unsqueeze(2).repeat(1, 1, self.embed_size).\
            float().to(self.device).detach()
        #print('mask shape',mask.shape)
        bs=x.shape[0]
        

        # trans推理获得最后一层表示decoder的表示
        # 注意是否返回的是dict

        if self.model_type=='pegasus':
            outputs=self.paraphraser.generate(input_ids=x,\
                attention_mask=attention_mask,num_beams=1,do_sample=True,\
                    return_dict_in_generate=True,\
                        output_hidden_states=True)
            # 注意输出的是BeamSearch、SampleSearch、GreedySearch
        elif self.model_type=='t5':
            outputs=self.paraphraser.generate(input_ids=x,\
                attention_mask=attention_mask,do_sample=True,\
                    return_dict_in_generate=True,\
                            num_beams=1 ,top_k=10,\
                                max_length=128, early_stopping=True,\
                        output_hidden_states=True)
            # decoder_inputs=self.create_T5_decoder_bos(bs)
            # outputs=self.paraphraser(input_ids=x,decoder_input_ids=decoder_inputs,\
            #     return_dict=True,output_hidden_states=True)
        #decoder_hidden_states:(seq_len*(n_layers*([batch_size,1,hidde_dim])))
        # print(outputs.keys())
        # print(type(outputs))
        # print(outputs.sequences.size())
        # print('batch_zise: {}'.format(bs))
        decoder_hidden_states=outputs.decoder_hidden_states
        seq_len=len(decoder_hidden_states)
        # print(type(decoder_hidden_states))
        # print('len decoder_hidden_states',seq_len)
        # print('len num_ayer:',len(decoder_hidden_states[0]))
        last_hidden_states=[]
        # print(outputs.encoder_hidden_states[0][0].size())
        for token_idx in range(seq_len):
            last_hidden_states.append(decoder_hidden_states[token_idx][-1])
        hidden=torch.cat(last_hidden_states,dim=1)
            # (batch_size, sequence_length, hidden_size)
        #masked_hidden = hidden * mask
        # print('hidden_szie: ',hidden.size())
        avg_hidden = torch.mean(hidden,dim=1)
        # print('avg_hidden size: ',avg_hidden.size())
        return avg_hidden
    
    def forward(self,x):
        '''
        tokenize后的输入模型的数据（token_id等等）
        '''
        avg_representation=self.avg_representation(x.to(self.device))
        #获得分类器\log(p(a|x))分布
        logits = self.classifier(avg_representation)
        # print('logits size ',logits.size())
        probs = F.log_softmax(logits, dim=-1)
        # print('probs size ',probs.size())
        # exit()
        return probs
    
    def inference(self,input_sentence):
        input_t = self.tokenizer.encode(input_sentence)
        input_t = torch.tensor([input_t], dtype=torch.long, device=self.device)

        #self就是调用自己forward
        log_probs = self.forward(input_t).data.cpu().numpy().flatten().tolist()
        prob = [math.exp(log_prob) for log_prob in log_probs]
        return prob


    

