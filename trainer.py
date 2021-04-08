import torch
import transformers

def train(model, dataloader, loss_fn, optim, device, scheduler, n_examples):
    """
    trainer function
    """
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in dataloader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)

        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        optim.step()
        loss.zero_grad()
        loss.backward()
    
    return losses