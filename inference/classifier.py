import torch.nn as nn
'''
即插即用的分类器
'''

class ClassifierHead(nn.Module):
    def __init__(self,class_size, embed_size,hidden_size=128*2) -> None:
        super(ClassifierHead,self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.hidden_size=hidden_size

        if hidden_size is None:
            self.mpl = nn.Linear(embed_size, class_size)
        else:
            self.mpl=nn.Sequential(nn.Linear(embed_size, hidden_size),
                                    nn.ReLU(),nn.Linear(hidden_size, class_size))
    
    def forward(self,hidden_state):
        '''
        输入为Trans的最后一层的平均（Diffusion中则是对每个token的x_i进行预测）
        '''
        logits = self.mpl(hidden_state)
        return logits