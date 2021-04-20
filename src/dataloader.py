from src.datareader import datareader, PAD_INDEX
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
# import logging

# logger = logging.getLogger()

class Dataset(data.Dataset):
    def __init__(self, domain, label, X, y, max_len, tokenizer):
        self.domain = domain
        self.label = label
        self.X = X
        self.y = y
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.label[index],
            self.X[index],
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        ### BERT tokenizes sequence into subword tokens --> BIO labels should be modified to match with them
        y = []
        for i, word in enumerate(self.X[index].split()):
            tokenized = self.tokenizer.tokenize(word)
            num_subwords = len(tokenized)

            y.extend([self.y[index][i]] * num_subwords)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'bio_tag': y,
        }
    
    def __len__(self):
        return len(self.X)


def pad_tensor(dim0, dim1, tensor, front_pad=None):
    """
    tensor: 1-d List or 1-d List of torch.LongTensor
    """
    padded_tensor = torch.LongTensor(dim0, dim1).fill_(PAD_INDEX)
    for i, vector in enumerate(tensor):
        vector_len = len(vector)
        front = 0
        if front_pad is not None:
            front = front_pad[i] + 1

        padded_tensor[i, front:front+vector_len] = torch.LongTensor(vector)

    # print("{}\t{}\t{}".format(length, y_len, padded_y.shape))
    return padded_tensor


def collate_fn(data):
    """
    Data modification
    1. 

    """
    query, attention_mask, y, sep_appear = [],[],[],[]
    # query, attention_mask, y1, y2, sep_appear, sentence_range = [],[],[],[],[],[]
    for d in data:
        query.append(d['input_ids'])
        attention_mask.append(d['attention_mask'])
        y.append(d['bio_tag'])
        sep_appear.append((d['input_ids'] == 102).nonzero(as_tuple=True)[0][0])
        # sentence_range.append((d['input_ids'] == 102).nonzero(as_tuple=True)[0])

    # print(query, attention_mask, y1, y2)
    # print(query[1].shape, len(y1[1]))
    lengths = [len(q) for q in query] # length of query, attention mask is same
    max_len = max(lengths)

    padded_query = pad_tensor(len(query), max_len, query)
    padded_attention_mask = pad_tensor(len(attention_mask), max_len, attention_mask)
    padded_y = pad_tensor(len(y), max_len, y, sep_appear)
    
    return padded_query, padded_attention_mask, padded_y


def get_dataloader(tgt_domain, batch_size, n_samples, tokenizer):
    all_data, max_length = datareader()
    train_data = {"domain": [], "label": [], "utter": [], "y": []}
    for dm_name, dm_data in all_data.items():
        if dm_name != tgt_domain:
            train_data["domain"].extend(dm_data["domain"])
            train_data["label"].extend(dm_data["label"])
            train_data["utter"].extend(dm_data["utter"])
            train_data["y"].extend(dm_data["y"])

    val_data = {"domain": [], "label": [], "utter": [], "y": []}
    test_data = {"domain": [], "label": [], "utter": [], "y": []}
    if n_samples == 0:
        # first 500 samples as validation set
        val_data["domain"] = all_data[tgt_domain]["domain"][:500]  
        val_data["label"] = all_data[tgt_domain]["label"][:500]
        val_data["utter"] = all_data[tgt_domain]["utter"][:500]
        val_data["y"] = all_data[tgt_domain]["y"][:500]

        # the rest as test set
        test_data["domain"] = all_data[tgt_domain]["domain"][500:]    
        test_data["label"] = all_data[tgt_domain]["label"][500:]
        test_data["utter"] = all_data[tgt_domain]["utter"][500:]
        test_data["y"] = all_data[tgt_domain]["y"][500:]

    else:
        # first n samples as train set
        train_data["domain"].extend(all_data[tgt_domain]["domain"][:n_samples])
        train_data["label"].extend(all_data[tgt_domain]["label"][:n_samples])
        train_data["utter"].extend(all_data[tgt_domain]["utter"][:n_samples])
        train_data["y"].extend(all_data[tgt_domain]["y"][:n_samples])

        # from n to 500 samples as validation set
        val_data["domain"] = all_data[tgt_domain]["domain"][n_samples:500]  
        val_data["label"] = all_data[tgt_domain]["label"][n_samples:500]
        val_data["utter"] = all_data[tgt_domain]["utter"][n_samples:500]
        val_data["y"] = all_data[tgt_domain]["y"][n_samples:500]

        # the rest as test set (same as zero-shot)
        test_data["domain"] = all_data[tgt_domain]["domain"][500:]
        test_data["label"] = all_data[tgt_domain]["label"][500:]
        test_data["utter"] = all_data[tgt_domain]["utter"][500:]
        test_data["y"] = all_data[tgt_domain]["y"][500:]

    dataset_tr = Dataset(train_data["domain"], train_data["label"], train_data["utter"], train_data["y"], max_length, tokenizer)
    dataset_val = Dataset(val_data["domain"], val_data["label"], val_data["utter"], val_data["y"], max_length, tokenizer)
    dataset_test = Dataset(test_data["domain"], test_data["label"], test_data["utter"], test_data["y"], max_length, tokenizer)

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_tr, dataloader_val, dataloader_test