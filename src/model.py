from transformers import BertModel, BertConfig
from src.modules import CRF
import torch
from torch import nn

class TripletTagger(nn.Module):
    def __init__(self, bert_hidden_dim, num_binslot=3):
        super(TripletTagger, self).__init__()
        
        self.num_binslot = num_binslot
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
            prediction: logits of predictions
            crf_loss: loss of crf
        """
        prediction = self.linear(inputs)
        crf_loss = self.crf_layer.loss(prediction, y)
        return prediction, crf_loss
    
    def crf_decode(self, logits):
        """
        crf decode
        logits to labeling (0/1/2 == O/B/I)
        Input:
            logits: (bsz, max_seq_len, num_entity)
        Output:
            pred: (bsz, max_seq_len)
        """
        return torch.argmax(logits, dim=2)

class SlotFillingModel(nn.Module):
    def __init__(self, args):
        super(SlotFillingModel, self).__init__()

        # hyperparameter
        self.num_tags = args.num_tags

        # model
        bert_config = BertConfig.from_pretrained('bert-base-uncased', hidden_dropout_prob=args.dropout_rate)
        self.query_bert = BertModel.from_pretrained("bert-base-uncased", config=bert_config)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.classifier = nn.Linear(self.query_bert.config.hidden_size, self.num_tags)
        self.crf = CRF(self.num_tags)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")


    def forward(self, q_inputs):
        """
        parameters
        ----------
        q_inputs: consist of (tokenized (slot_desc, utterance) pairs, (gold) bio labels)
        """
        query_outputs = self.query_bert(
            q_inputs['input_ids'],
            q_inputs['attention_mask'],
            q_inputs['token_type_ids']
        )
        query_sequence_outputs = query_outputs.last_hidden_state
        query_cls_output = query_sequence_outputs[:, 0, :]

        # BIO classification
        # outputs = self.dropout(query_sequence_outputs)
        logits = self.classifier(query_sequence_outputs)
        crf_loss = self.crf.loss(logits, q_inputs['labels'])

        return crf_loss, logits
