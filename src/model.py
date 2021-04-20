from src.modules import CRF
import torch
from torch import nn
# from torchcrf import CRF

class TripletTagger(nn.Module):
    def __init__(self, bert_hidden_dim):
        super(TripletTagger, self).__init__()
        
        self.num_binslot = 3
        self.hidden_dim = bert_hidden_dim
        self.linear = nn.Linear(self.hidden_dim, self.num_binslot)
        self.crf_layer = CRF(self.num_binslot)
        
    def forward(self, inputs, y):
        """ create crf loss
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.linear(inputs)
        crf_loss = self.crf_layer.loss(prediction, y)

        return prediction, crf_loss
    
    def crf_decode(self, logits):
        """
        crf decode
        0/1/2 --> O/B/I
        Input:
            logits: (bsz, max_seq_len, num_entity)
        Output:
            pred: (bsz, max_seq_len)
        """
        return torch.argmax(logits, dim=2)
