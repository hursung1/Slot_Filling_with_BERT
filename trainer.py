import torch
from tqdm import tqdm
from src.modules import f1_score

def train(model, tagger, dataloader, optim, device, scheduler):
    """
    trainer function
    """
    model.train()
    tagger.train()
    
    pbar = tqdm(dataloader, total=len(dataloader))
    losses = []
    acc_list = []
    for (query, attention_mask, targets) in pbar:
        total = 0
        correct = 0
        optim.zero_grad()
        total += query.shape[0] * query.shape[1]
        if device:
            query = query.cuda()
            attention_mask = attention_mask.cuda()
            targets = targets.cuda()

        model_outputs = model(query, attention_mask)
        last_hidden_state = model_outputs.last_hidden_state
        
        logits, loss = tagger(last_hidden_state, targets)
        pred = tagger.crf_decode(logits)
        correct += (pred == targets).sum().detach().cpu().item()

        losses.append(loss.detach().cpu().item())
        loss.backward()

        optim.step()
        scheduler.step()

        pbar.set_description(f"LOSS: {losses[-1]:.3f} ACC: {correct/total:.3f}")
        acc_list.append(correct/total)

        del query
        del attention_mask
        del targets

    torch.cuda.empty_cache()
    avg_acc = sum(acc_list) / len(acc_list)
    return losses, avg_acc


def eval(model, tagger, tokenizer, dataloader, device, is_test=False):
    """
    evalutation function for validation dataset and test dataset
    
    returns
    ----------
    List of losses, F1 Score
    """
    model.eval()
    tagger.eval()
    losses = []
    total_preds = []
    total_targets = []

    total = 0
    correct = 0
    with torch.no_grad():
        for (query, attention_mask, targets) in dataloader:
            if device:
                query = query.cuda()
                attention_mask = attention_mask.cuda()
                targets = targets.cuda()

            model_outputs = model(query, attention_mask)
            last_hidden_state = model_outputs.last_hidden_state
            logits, loss = tagger(last_hidden_state, targets)
            losses.append(loss.detach().cpu().item())
            
            pred = tagger.crf_decode(logits)
            total += query.shape[0] * query.shape[1]
            correct += (pred == targets).sum().detach().cpu().item()

            total_preds.append(pred.flatten().tolist())
            total_targets.append(targets.flatten().tolist())
        
        rand = torch.randint(query.size()[0], (1,)).item()
        decoded = tokenizer.decode(query[rand])

        print("Query     : ", decoded)
        print("Answer    : ", targets[rand])
        print("Prediction: ", pred[rand])
        
        f1 = f1_score(total_preds, total_targets)
        torch.cuda.empty_cache()

    acc = correct / total
    return losses, acc, f1