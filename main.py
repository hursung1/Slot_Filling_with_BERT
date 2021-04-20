import torch
from torch import nn, optim
import matplotlib.pyplot as plt

# import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from transformers import pipeline, BertModel, BertTokenizer, BertConfig
from transformers.utils.dummy_pt_objects import BertForTokenClassification, PretrainedBartModel

from src.utils import init_experiment
from src.dataloader import get_dataloader
from src.model import TripletTagger
from config import get_params
from trainer import train, eval
import os

def main(params):
    if not os.path.isdir('./output_fig'):
        os.mkdir('./output_fig')
    
    if not os.path.isdir('./out_log'):
        os.mkdir('./out_log')
    
    logfile = open(f'./out_log/logfile_{params.tgt_dm}_sample{params.n_samples}_epoch{params.epoch}', 'w')

    # initialize experiment
    print(f'Target Domain: {params.tgt_dm}\tN Samples: {params.n_samples}\tEpochs: {params.epoch}')
    logfile.write(f'Target Domain: {params.tgt_dm}\tN Samples: {params.n_samples}\tEpochs: {params.epoch}\n')

    # set default device
    cuda_available = True if torch.cuda.is_available() else False

    # load pretrained BERT and define model 
    if params.use_plain:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    else:
        tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')
        model = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')
    
    BIOTagger = TripletTagger(model.config.hidden_size)
    if cuda_available:
        model = nn.DataParallel(model.cuda())
        BIOTagger = BIOTagger.cuda()

    # get dataloader
    dataloader_tr, dataloader_val, dataloader_test = get_dataloader(params.tgt_dm, params.batch_size, params.n_samples, tokenizer)

    # nlp = pipeline('ner', model=model, tokenizer=tokenizer)

    # loss function, optimizer, ...
    model_parameters = [
        {"params": model.parameters()},
        {"params": BIOTagger.parameters()}
    ]
    optim = AdamW(model_parameters, lr=params.lr, correct_bias=False)

    total_steps = params.epoch * len(dataloader_tr)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)
    
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    val_f1_list = []
    xlabel = [i for i in range(1, params.epoch + 1)]
    
    val_losses, val_acc, val_f1 = eval(model, BIOTagger, tokenizer, dataloader_val, cuda_available, False)
    print(f"Before Training\nVal Loss {sum(val_losses)/len(val_losses):.3f}\tVal Accuracy {val_acc:.3f}\tF1 Score {val_f1}")
    logfile.write(f"Before Training\nVal Loss {sum(val_losses)/len(val_losses):.3f}\tVal Accuracy {val_acc:.3f}\tF1 Score {val_f1}\n")

    for e in range(params.epoch):
        print(f"Training... EPOCH {e+1}")
        tr_losses, tr_acc = train(model, BIOTagger, dataloader_tr, optim, cuda_available, scheduler)
        
        print(f"Validation... EPOCH {e+1}")
        val_losses, val_acc, val_f1 = eval(model, BIOTagger, tokenizer, dataloader_val, cuda_available, False)
        
        print(f'EPOCH {e+1}\tVal Loss {sum(val_losses)/len(val_losses):.3f}\tVal Accuracy {val_acc:.3f}\tF1 Score {val_f1}')
        logfile.write(f'EPOCH {e+1}\tVal Loss {sum(val_losses)/len(val_losses):.3f}\tVal Accuracy {val_acc:.3f}\tF1 Score {val_f1}\n')

        train_loss_list.extend(tr_losses)
        val_loss_list.extend(val_losses)
        train_acc_list.append(tr_acc)
        val_acc_list.append(val_acc)
        val_f1_list.append(val_f1)

    # train loss 
    plt.plot(train_loss_list)
    plt.title('Train Set Loss')
    plt.xlabel('Iterations')
    plt.ylabel('loss')
    plt.savefig(f'output_fig/{params.tgt_dm}_{params.n_samples}_trainloss.png', dpi=500)
    plt.close()

    # val loss 
    plt.figure()
    plt.plot(val_loss_list)
    plt.title('Validation Set Loss')
    plt.xlabel('Iterations')
    plt.ylabel('loss')
    plt.savefig(f'output_fig/{params.tgt_dm}_{params.n_samples}_valloss.png', dpi=500)
    plt.close()
    
    # val accuracy 
    plt.figure()
    plt.plot(xlabel, val_acc_list)
    plt.title('Validation Set Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('validation set accuracy')
    plt.savefig(f'output_fig/{params.tgt_dm}_{params.n_samples}_valacc.png', dpi=500)
    plt.close()

    # val f1
    plt.figure()
    plt.plot(xlabel, val_f1_list)
    plt.title('Validation Set Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('validation set accuracy')
    plt.savefig(f'output_fig/{params.tgt_dm}_{params.n_samples}_valf1.png', dpi=500)
    plt.close()
    
    # test
    test_losses, test_acc, test_f1 = eval(model, BIOTagger, tokenizer, dataloader_test, cuda_available, True)
    avg_test_loss = sum(test_losses)/len(test_losses)

    print(f"Test\nLoss: {avg_test_loss:.3f}\tAccuracy: {test_acc:.3f}\tF1 Score: {test_f1:.3f}")
    logfile.write(f"Test\nLoss: {avg_test_loss:.3f}\tAccuracy: {test_acc:.3f}\tF1 Score: {test_f1:.3f}")
    
    logfile.close()


if __name__=="__main__":
    params = get_params()
    main(params)