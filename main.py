### for test
import torch
from torch import nn, optim

import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from transformers import pipeline, BertForSequenceClassification, BertTokenizer

from src.utils import init_experiment
from src.dataloader import get_dataloader
from config import get_params
from trainer import train

from tqdm import tqdm
import os

def main(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    
    # load BERT pretrained model
    if params.use_plain:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    else:
        tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')
        model = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')

    # get dataloader
    dataloader_tr, dataloader_val, dataloader_test, vocab = get_dataloader(params.tgt_dm, params.batch_size, params.n_samples, tokenizer)
    # set default device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    os.environ["CUDA_VISIBLE_DEVICES"] = params.cuda_device

    
    # nlp = pipeline('ner', model=model, tokenizer=tokenizer)

    # loss function, optimizer, ...
    loss_fn = nn.CrossEntropyLoss().to(device)
    optim = AdamW(model.parameters(), lr=1e-5, correct_bias=False)

    total_steps = params.epoch * len(dataloader_tr)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)
    
    for e in range(params.epoch):
        train(model, dataloader_tr, loss_fn, optim, device, scheduler, params.n_samples)
        # logger.info("============== epoch {} ==============".format(e+1))
        # loss_bin_list, loss_slotname_list = [], []
        # pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))

        # for i, (query, lengths, y_bin, y_final, y_dm) in pbar:
        #     query, lengths, y_bin, y_final, y_dm = query.to(device), lengths.to(device), y_bin.to(device), y_final.to(device), y_dm.to(device)
        #     loss_bin, loss_slotname = train(query, lengths, y_bin, y_final, y_dm)


if __name__=="__main__":
    params = get_params()
    main(params)