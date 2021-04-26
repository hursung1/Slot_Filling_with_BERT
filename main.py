import torch
from torch import nn, optim
import matplotlib.pyplot as plt

# import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from transformers import pipeline, BertModel, BertTokenizer, BertConfig
from transformers.utils.dummy_pt_objects import BertForTokenClassification

from src.utils import save_plot
from src.dataloader import get_dataloader
from src.model import TripletTagger
from config import get_params
from trainer import train, eval
import os

def main(params):

    # set default device
    cuda_available = True if torch.cuda.is_available() else False

    # load pretrained BERT and define model 
    # print(params.use_plain)
    if params.use_plain:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        logfile_dir_path = f'./out_log/Plain/{params.tgt_dm}/Sample{params.n_samples}/'
        figure_dir_path = f'./output_fig/Plain/{params.tgt_dm}/Sample{params.n_samples}/'
        bert_model_save_path = f'./experiments/Bert/Plain/{params.tgt_dm}/Sample{params.n_samples}/'
        tagger_model_save_path = f'./experiments/Tagger/Plain/{params.tgt_dm}/Sample{params.n_samples}/'

    else:
        tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')
        model = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')
        logfile_dir_path = f'./out_log/NER_Pretrained/{params.tgt_dm}/Sample{params.n_samples}/'
        figure_dir_path = f'./output_fig/NER_Pretrained/{params.tgt_dm}/Sample{params.n_samples}/'
        bert_model_save_path = f'./experiments/Bert/NER_Pretrained/{params.tgt_dm}/Sample{params.n_samples}/'
        tagger_model_save_path = f'./experiments/Tagger/NER_Pretrained/{params.tgt_dm}/Sample{params.n_samples}/'

    BIOTagger = TripletTagger(model.config.hidden_size)
    if cuda_available:
        model = nn.DataParallel(model.cuda())
        BIOTagger = BIOTagger.cuda()

    # get dataloader
    dataloader_tr, dataloader_val, _ = get_dataloader(params.tgt_dm, params.batch_size, params.n_samples, tokenizer)

    # loss function, optimizer, ...
    model_parameters = [
        {"params": model.parameters()},
        {"params": BIOTagger.parameters()}
    ]
    optim = AdamW(model_parameters, lr=params.lr, correct_bias=False)

    total_steps = params.epoch * len(dataloader_tr)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)

    os.makedirs(logfile_dir_path, exist_ok=True)
    os.makedirs(figure_dir_path, exist_ok=True)
    os.makedirs(bert_model_save_path, exist_ok=True)
    os.makedirs(tagger_model_save_path, exist_ok=True)
    
    logfile = open(logfile_dir_path + f'logfile_{params.tgt_dm}_sample{params.n_samples}.txt', 'w')
    print(f'Target Domain: {params.tgt_dm}\tN Samples: {params.n_samples}')
    logfile.write(f'Target Domain: {params.tgt_dm}\tN Samples: {params.n_samples}\n')

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    val_f1_list = []
    xlabel = [i for i in range(1, params.epoch + 1)]
    max_val_f1 = 0
    best_model_counter = 0
    e = 0
    
    val_losses, val_acc, val_f1 = eval(model, BIOTagger, tokenizer, dataloader_val, cuda_available, False)
    print(f"Before Training\nVal Loss {sum(val_losses)/len(val_losses):.3f}\tVal Accuracy {val_acc:.3f}\tF1 Score {val_f1}")
    logfile.write(f"Before Training\nVal Loss {sum(val_losses)/len(val_losses):.3f}\tVal Accuracy {val_acc:.3f}\tF1 Score {val_f1}\n")

    while best_model_counter < 5:
        print(f"Training... EPOCH {e+1}")
        tr_losses, tr_acc = train(model, BIOTagger, dataloader_tr, optim, cuda_available, scheduler)
        
        print(f"Validation... EPOCH {e+1}")
        val_losses, val_acc, val_f1 = eval(model, BIOTagger, tokenizer, dataloader_val, cuda_available, False)
        
        print(f'EPOCH {e+1}\tVal Loss {sum(val_losses)/len(val_losses):.3f}\tVal Accuracy {val_acc:.3f}\tF1 Score {val_f1}')
        logfile.write(f'EPOCH {e+1}\tVal Loss {sum(val_losses)/len(val_losses):.3f}\tVal Accuracy {val_acc:.3f}\tF1 Score {val_f1}\n')

        # save model which shows best validation f1 score
        best_model_counter += 1
        e += 1
        if val_f1 > max_val_f1:
            print("Found Better Model!")
            logfile.write("Found Better Model!\n")
            tokenizer.save_pretrained(bert_model_save_path)
            model.module.save_pretrained(bert_model_save_path)
            torch.save(BIOTagger.state_dict(), tagger_model_save_path+'state_dict_model.pt')
            max_val_f1 = val_f1
            best_model_counter = 0

        train_loss_list.extend(tr_losses)
        val_loss_list.extend(val_losses)
        train_acc_list.append(tr_acc)
        val_acc_list.append(val_acc)
        val_f1_list.append(val_f1)

    save_plot(
        'Train Set Loss',
        'Iterations',
        'loss',
        figure_dir_path + f'{params.tgt_dm}_{params.n_samples}_trainloss.png',
        train_loss_list
    )
    save_plot(
        'Validation Set Loss',
        'Iterations',
        'loss',
        figure_dir_path + f'{params.tgt_dm}_{params.n_samples}_valloss.png',
        val_loss_list
    )
    save_plot(
        'Validation Set Accuracy',
        'epoch',
        'accuracy',
        figure_dir_path + f'{params.tgt_dm}_{params.n_samples}_valacc.png',
        val_acc_list,
        xlabel
    )
    save_plot(
        'Validation Set F1-Score',
        'epoch',
        'f1-score',
        figure_dir_path + f'{params.tgt_dm}_{params.n_samples}_valf1.png',
        val_f1_list,
        xlabel
    )

    logfile.close()


def test(params):
    # test for seen / unseen labeled data
    cuda_available = False
    if torch.cuda.is_available():
        cuda_available = True

    if params.use_plain:
        bert_model_save_path = f'./experiments/Bert/Plain/{params.tgt_dm}/Sample{params.n_samples}/'
        tagger_model_save_path = f'./experiments/Tagger/Plain/{params.tgt_dm}/Sample{params.n_samples}/state_dict_model.pt'
        tokenizer = BertTokenizer.from_pretrained(bert_model_save_path)
        model = BertModel.from_pretrained(bert_model_save_path)

    else:
        bert_model_save_path = f'./experiments/Bert/NER_Pretrained/{params.tgt_dm}/Sample{params.n_samples}/'
        tagger_model_save_path = f'./experiments/Tagger/NER_Pretrained/{params.tgt_dm}/Sample{params.n_samples}/state_dict_model.pt'
        tokenizer = AutoTokenizer.from_pretrained(bert_model_save_path)
        model = AutoModelForTokenClassification.from_pretrained(bert_model_save_path)
        
    BIOTagger = TripletTagger(model.config.hidden_size)
    BIOTagger.load_state_dict(torch.load(tagger_model_save_path))

    if cuda_available:
        model = model.cuda()
        BIOTagger = BIOTagger.cuda()

    _, _, dataloader_test = get_dataloader(params.tgt_dm, params.batch_size, params.n_samples, tokenizer)

    test_losses, test_acc, test_f1 = eval(model, BIOTagger, tokenizer, dataloader_test, cuda_available, True)
    avg_test_loss = sum(test_losses)/len(test_losses)

    print(f"Test\nLoss: {avg_test_loss:.3f}\tAccuracy: {test_acc:.3f}\tF1 Score: {test_f1:.3f}")
    

if __name__=="__main__":
    params = get_params()
    if len(params.model_path) == 0:
        main(params)
    else:
        test(params)