from src.datareader import datareader, PAD_INDEX, domain2slot
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

class Dataset(data.Dataset):
    def __init__(self, domain, label, X, y, max_len, tokenizer):
        self.domain = domain
        self.label = label
        self.X = X
        self.y = y
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        slot = self.label[index].split()
        utter = self.X[index].split()
        encoding = self.tokenizer(slot, utter, is_split_into_words=True)
        ### BERT tokenizes sequence into subword tokens --> BIO labels should be modified to match with them
        subword_ids = encoding.word_ids()
        labels = self.y[index]
        new_labels = []
        
        none_counter = 0
        for i, word_idx in enumerate(subword_ids):
            if none_counter < 2 or word_idx is None:
                new_labels.append(0)
                if word_idx is None:
                    none_counter += 1
            
            elif none_counter == 2:
                new_labels.append(labels[word_idx])

        # y = []
        # for i, word in enumerate(self.X[index].split()):
        #     tokenized = self.tokenizer.tokenize(word)
        #     num_subwords = len(tokenized)

        #     y.extend([self.y[index][i]] * num_subwords)

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids'],
            'labels': new_labels,
        }
    
    def __len__(self):
        return len(self.X)


def pad_tensor(features, max_seq_len):
# def pad_tensor(dim0, dim1, tensor, front_pad=None):
    """
    features: list of lists, each element equals to output of hf tokenizer
    """
    padded_features = []
    for f in features:
        original_value_len = len(f)
        for _ in range(original_value_len, max_seq_len):
            f.append(PAD_INDEX)
        
        padded_features.append(f)

    return torch.tensor(padded_features, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.tensor(padded_features, dtype=torch.long)


def collate_fn(features):
    """
    Collate function for SLU Model
    pad at right side
    """
    padded_features = {}
    batch_size = len(features)
    max_seq_len = 0

    _batch = {}
    for f in features:
        for k, v in f.items():
            try:
                padded_features[k].append(v)
            except:
                padded_features[k] = [v]
        
            feature_len = len(v)
            if feature_len > max_seq_len:
                max_seq_len = feature_len

    for k, v in padded_features.items():
        v = pad_tensor(v, max_seq_len)
        padded_features[k] = v


    return padded_features


def get_dataloader(tgt_domain, batch_size, n_samples, data_path, tokenizer):
    all_data, max_length = datareader(data_path)
    train_data = {"domain": [], "label": [], "utter": [], "y": []}
    for dm_name, dm_data in all_data.items():
        if dm_name != tgt_domain:
            train_data["domain"].extend(dm_data["domain"])
            train_data["label"].extend(dm_data["label"])
            train_data["utter"].extend(dm_data["utter"])
            train_data["y"].extend(dm_data["y"])

    val_data = {"domain": [], "label": [], "utter": [], "y": []}
    test_data = {"domain": [], "label": [], "utter": [], "y": []}

    num_tgt_slots = len(domain2slot[tgt_domain])
    val_split = 500*num_tgt_slots # validation: 500 utterances
    train_split = n_samples * num_tgt_slots
    
    if n_samples == 0:
        # first 500 samples as validation set
        val_data["domain"] = all_data[tgt_domain]["domain"][:val_split]  
        val_data["label"] = all_data[tgt_domain]["label"][:val_split]
        val_data["utter"] = all_data[tgt_domain]["utter"][:val_split]
        val_data["y"] = all_data[tgt_domain]["y"][:val_split]

        # the rest as test set
        test_data["domain"] = all_data[tgt_domain]["domain"][val_split:]    
        test_data["label"] = all_data[tgt_domain]["label"][val_split:]
        test_data["utter"] = all_data[tgt_domain]["utter"][val_split:]
        test_data["y"] = all_data[tgt_domain]["y"][val_split:]

    else:
        # first n samples as train set
        train_data["domain"].extend(all_data[tgt_domain]["domain"][:train_split])
        train_data["label"].extend(all_data[tgt_domain]["label"][:train_split])
        train_data["utter"].extend(all_data[tgt_domain]["utter"][:train_split])
        train_data["y"].extend(all_data[tgt_domain]["y"][:train_split])

        # from n to 500 samples as validation set
        val_data["domain"] = all_data[tgt_domain]["domain"][train_split:val_split]  
        val_data["label"] = all_data[tgt_domain]["label"][train_split:val_split]
        val_data["utter"] = all_data[tgt_domain]["utter"][train_split:val_split]
        val_data["y"] = all_data[tgt_domain]["y"][train_split:val_split]

        # the rest as test set (same as zero-shot)
        test_data["domain"] = all_data[tgt_domain]["domain"][val_split:]
        test_data["label"] = all_data[tgt_domain]["label"][val_split:]
        test_data["utter"] = all_data[tgt_domain]["utter"][val_split:]
        test_data["y"] = all_data[tgt_domain]["y"][val_split:]

    dataset_tr = Dataset(train_data["domain"], train_data["label"], train_data["utter"], train_data["y"], max_length, tokenizer)
    dataset_val = Dataset(val_data["domain"], val_data["label"], val_data["utter"], val_data["y"], max_length, tokenizer)
    dataset_test = Dataset(test_data["domain"], test_data["label"], test_data["utter"], test_data["y"], max_length, tokenizer)

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=len(domain2slot[tgt_domain]), shuffle=False, collate_fn=collate_fn)

    return dataloader_tr, dataloader_val, dataloader_test