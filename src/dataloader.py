
from src.datareader import datareader, PAD_INDEX
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger()


class Dataset(data.Dataset):
    def __init__(self, X, y1, y2, max_len, tokenizer):
        self.X = X
        self.y1 = y1 
        self.y2 = y2
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        encoding = self.tokenizer.encode_plus(
            self.X[index],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # 'query': self.X[index],
            'bio': self.y1[index],
            'bio_with_label': self.y2[index],
            'domains': self.domains[index],
        }
    
    def __len__(self):
        return len(self.X)


def collate_fn(data):
    # print(data)
    input_ids, attention_mask, X, y1, y2, domains = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    domains = torch.LongTensor(domains)
    
    return padded_seqs, attention_mask, lengths, y1, y2, domains, input_ids


def get_dataloader(tgt_domain, batch_size, n_samples, tokenizer):
    all_data, vocab, max_length = datareader()
    train_data = {"domain_utter": [], "y1": [], "y2": []}
    for dm_name, dm_data in all_data.items():
        if dm_name != tgt_domain:
            train_data["domain_utter"].extend(dm_data["domain_utter"])
            train_data["y1"].extend(dm_data["y1"])
            train_data["y2"].extend(dm_data["y2"])

    val_data = {"domain_utter": [], "y1": [], "y2": [], "domains": []}
    test_data = {"domain_utter": [], "y1": [], "y2": [], "domains": []}
    if n_samples == 0:
        # first 500 samples as validation set
        val_data["domain_utter"] = all_data[tgt_domain]["domain_utter"][:500]  
        val_data["y1"] = all_data[tgt_domain]["y1"][:500]
        val_data["y2"] = all_data[tgt_domain]["y2"][:500]
        val_data["domains"] = all_data[tgt_domain]["domains"][:500]

        # the rest as test set
        test_data["domain_utter"] = all_data[tgt_domain]["domain_utter"][500:]    
        test_data["y1"] = all_data[tgt_domain]["y1"][500:]      # rest as test set
        test_data["y2"] = all_data[tgt_domain]["y2"][500:]      # rest as test set
        test_data["domains"] = all_data[tgt_domain]["domains"][500:]    # rest as test set

    else:
        # first n samples as train set
        train_data["domain_utter"].extend(all_data[tgt_domain]["domain_utter"][:n_samples])
        train_data["y1"].extend(all_data[tgt_domain]["y1"][:n_samples])
        train_data["y2"].extend(all_data[tgt_domain]["y2"][:n_samples])
        train_data["domains"].extend(all_data[tgt_domain]["domains"][:n_samples])

        # from n to 500 samples as validation set
        val_data["domain_utter"] = all_data[tgt_domain]["domain_utter"][n_samples:500]  
        val_data["y1"] = all_data[tgt_domain]["y1"][n_samples:500]
        val_data["y2"] = all_data[tgt_domain]["y2"][n_samples:500]
        val_data["domains"] = all_data[tgt_domain]["domains"][n_samples:500]

        # the rest as test set (same as zero-shot)
        test_data["domain_utter"] = all_data[tgt_domain]["domain_utter"][500:]
        test_data["y1"] = all_data[tgt_domain]["y1"][500:]
        test_data["y2"] = all_data[tgt_domain]["y2"][500:]
        test_data["domains"] = all_data[tgt_domain]["domains"][500:]

    dataset_tr = Dataset(train_data["domain_utter"], train_data["y1"], train_data["y2"], max_length, tokenizer)
    dataset_val = Dataset(val_data["domain_utter"], val_data["y1"], val_data["y2"], max_length, tokenizer)
    dataset_test = Dataset(test_data["domain_utter"], test_data["y1"], test_data["y2"], max_length, tokenizer)

    # dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader_tr, dataloader_val, dataloader_test, vocab