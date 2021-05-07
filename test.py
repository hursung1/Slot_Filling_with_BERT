from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForTokenClassification

from src.datareader import read_file, data_binarize
from src.dataloader import get_dataloader, Dataset, DataLoader, collate_fn
from src.model import TripletTagger
from config import get_params
from trainer import eval
import torch


def test(params, model, BIOTagger, tokenizer, cuda_available):
    f = open(f"test_all_target_{params.tgt_dm}_sample_{params.n_samples}", "w")
    print(f'Target Domain: {params.tgt_dm}\tN Samples: {params.n_samples}\tEpochs: {params.epoch}')
    f.write(f'Target Domain: {params.tgt_dm}\tN Samples: {params.n_samples}\n')
    # get dataloader
    _, _, dataloader_test = get_dataloader(params.tgt_dm, params.batch_size, params.n_samples, tokenizer)
    
    test_losses, test_acc, test_f1 = eval(model, BIOTagger, tokenizer, dataloader_test, cuda_available, True, f)
    avg_test_loss = sum(test_losses)/len(test_losses)

    print(f"Test\nLoss: {avg_test_loss:.4f}\tAccuracy: {test_acc:.4f}\tF1 Score: {test_f1:.4f}")
    f.write(f"Test\nLoss: {avg_test_loss:.4f}\tAccuracy: {test_acc:.4f}\tF1 Score: {test_f1:.4f}\n")
    f.close()

def test_on_seen_and_unseen(params, model, BIOTagger, tokenizer, cuda_available):
    f = open(f"test_seen_unseen_target_{params.tgt_dm}_sample_{params.n_samples}", "w")
    print(f'Target Domain: {params.tgt_dm}\tN Samples: {params.n_samples}\tEpochs: {params.epoch}')

    # read seen and unseen data
    print("Processing Unseen and Seen samples in %s domain ..." % params.tgt_dm)
    f.write(f"Processing Unseen and Seen samples in {params.tgt_dm} domain ...\n")
    unseen_data, unseen_max_len = read_file("data/"+params.tgt_dm+"/unseen_slots.txt", params.tgt_dm)
    seen_data, seen_max_len= read_file("data/"+params.tgt_dm+"/seen_slots.txt", params.tgt_dm)

    print("Binarizing data ...")
    if len(unseen_data["utter"]) > 0:
        unseen_data_bin = data_binarize(unseen_data)
    else:
        unseen_data_bin = None
    
    if len(seen_data["utter"]) > 0:
        seen_data_bin = data_binarize(seen_data)
    else:
        seen_data_bin = None

    print("Prepare dataloader ...")
    if unseen_data_bin:
        unseen_dataset = Dataset(unseen_data_bin["domain"], unseen_data_bin["label"], unseen_data_bin["utter"], unseen_data_bin["y"], unseen_max_len, tokenizer)

        unseen_dataloader = DataLoader(dataset=unseen_dataset, batch_size=params.batch_size, collate_fn=collate_fn, shuffle=False)

        _, _, test_f1 = eval(model, BIOTagger, tokenizer, unseen_dataloader, cuda_available, True, f)

        # _, f1_score, _ = slu_trainer.evaluate(unseen_dataloader, istestset=True)
        print(f"Unseen slots: Final slot F1 score: {test_f1:.4f}.")
        f.write(f"Unseen slots: Final slot F1 score: {test_f1:.4f}.\n")

    else:
        print("Number of unseen sample is zero")
        f.write("Number of unseen sample is zero\n")

    if seen_data_bin:
        seen_dataset = Dataset(seen_data_bin["domain"], seen_data_bin["label"], seen_data_bin["utter"], seen_data_bin["y"], seen_max_len, tokenizer)
        
        seen_dataloader = DataLoader(dataset=seen_dataset, batch_size=params.batch_size, collate_fn=collate_fn, shuffle=False)

        _, _, test_f1 = eval(model, BIOTagger, tokenizer, seen_dataloader, cuda_available, True, f)

        # _, f1_score, _ = slu_trainer.evaluate(seen_dataloader, istestset=True)
        print(f"Seen slots: Final slot F1 score: {test_f1:.4f}.")
        f.write(f"Seen slots: Final slot F1 score: {test_f1:.4f}.\n")

    else:
        print("Number of seen sample is zero")
        f.write("Number of seen sample is zero\n")

    f.close()


if __name__ == "__main__":
    params = get_params()

    cuda_available = False
    if torch.cuda.is_available():
        cuda_available = True

    if params.use_plain:
        bert_model_path = f'./experiments/Bert/Plain/{params.tgt_dm}/Sample{params.n_samples}/'
        tagger_model_path = f'./experiments/Tagger/Plain/{params.tgt_dm}/Sample{params.n_samples}/state_dict_model.pt'
        tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        model = BertModel.from_pretrained(bert_model_path)

    else:
        bert_model_path = f'./experiments/Bert/NER_Pretrained/{params.tgt_dm}/Sample{params.n_samples}/'
        tagger_model_path = f'./experiments/Tagger/NER_Pretrained/{params.tgt_dm}/Sample{params.n_samples}/state_dict_model.pt'
        tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        model = AutoModelForTokenClassification.from_pretrained(bert_model_path)

    BIOTagger = TripletTagger(model.config.hidden_size)
    BIOTagger.load_state_dict(torch.load(tagger_model_path))
    
    if cuda_available:
        model = model.cuda()
        BIOTagger = BIOTagger.cuda()

    if params.test_mode == "testset":
        test(params, model, BIOTagger, tokenizer, cuda_available)
    elif params.test_mode == "seen_unseen":
        test_on_seen_and_unseen(params, model, BIOTagger, tokenizer, cuda_available)
