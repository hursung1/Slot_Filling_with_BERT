
from src.datareader import datareader, PAD_INDEX
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import logging

BIO_dict = {'B': 1, 'I': 2, 'O': 0}

logger = logging.getLogger()

class Dataset(data.Dataset):
    def __init__(self, domain, X, y1, y2, max_len, tokenizer):
        self.domain = domain
        self.X = X
        self.y1 = y1 
        self.y2 = y2
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.domain[index],
            self.X[index],
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors='pt'
        )
        print(encoding['input_ids'].shape)
        # print(self.y1[index].shape)
        # print(self.y2[index].shape)

        ### BIO -> 1, 2, 0으로

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'bio': torch.tensor(self.y1[index], dtype=torch.long),
            'bio_with_label': torch.tensor(self.y2[index], dtype=torch.long),
        }
    
    def __len__(self):
        return len(self.X)


def get_dataloader(tgt_domain, batch_size, n_samples, tokenizer):
    all_data, vocab, max_length = datareader()
    train_data = {"domain": [], "domain_utter": [], "y1": [], "y2": []}
    for dm_name, dm_data in all_data.items():
        if dm_name != tgt_domain:
            train_data["domain_utter"].extend(dm_data["domain_utter"])
            train_data["y1"].extend(dm_data["y1"])
            train_data["y2"].extend(dm_data["y2"])
            train_data["domain"].extend(dm_data["domain"])

    val_data = {"domain": [], "domain_utter": [], "y1": [], "y2": []}
    test_data = {"domain": [], "domain_utter": [], "y1": [], "y2": []}
    if n_samples == 0:
        # first 500 samples as validation set
        val_data["domain_utter"] = all_data[tgt_domain]["domain_utter"][:500]  
        val_data["y1"] = all_data[tgt_domain]["y1"][:500]
        val_data["y2"] = all_data[tgt_domain]["y2"][:500]
        val_data["domain"] = all_data[tgt_domain]["domain"][:500]

        # the rest as test set
        test_data["domain_utter"] = all_data[tgt_domain]["domain_utter"][500:]    
        test_data["y1"] = all_data[tgt_domain]["y1"][500:]      # rest as test set
        test_data["y2"] = all_data[tgt_domain]["y2"][500:]      # rest as test set
        test_data["domain"] = all_data[tgt_domain]["domain"][500:]    # rest as test set

    else:
        # first n samples as train set
        train_data["domain_utter"].extend(all_data[tgt_domain]["domain_utter"][:n_samples])
        train_data["y1"].extend(all_data[tgt_domain]["y1"][:n_samples])
        train_data["y2"].extend(all_data[tgt_domain]["y2"][:n_samples])
        train_data["domain"].extend(all_data[tgt_domain]["domain"][:n_samples])

        # from n to 500 samples as validation set
        val_data["domain_utter"] = all_data[tgt_domain]["domain_utter"][n_samples:500]  
        val_data["y1"] = all_data[tgt_domain]["y1"][n_samples:500]
        val_data["y2"] = all_data[tgt_domain]["y2"][n_samples:500]
        val_data["domain"] = all_data[tgt_domain]["domain"][n_samples:500]

        # the rest as test set (same as zero-shot)
        test_data["domain_utter"] = all_data[tgt_domain]["domain_utter"][500:]
        test_data["y1"] = all_data[tgt_domain]["y1"][500:]
        test_data["y2"] = all_data[tgt_domain]["y2"][500:]
        test_data["domain"] = all_data[tgt_domain]["domain"][500:]

    dataset_tr = Dataset(train_data["domain"], train_data["domain_utter"], train_data["y1"], train_data["y2"], max_length, tokenizer)
    dataset_val = Dataset(val_data["domain"], val_data["domain_utter"], val_data["y1"], val_data["y2"], max_length, tokenizer)
    dataset_test = Dataset(test_data["domain"], test_data["domain_utter"], test_data["y1"], test_data["y2"], max_length, tokenizer)

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader_tr, dataloader_val, dataloader_test, vocab