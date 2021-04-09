import torch
import transformers

def train(model, dataloader, loss_fn, optim, device, scheduler, n_examples):
    """
    trainer function
    """
    model.train()
    losses = []
    correct_predictions = 0
    for d in dataloader:
        optim.zero_grad()
        loss.zero_grad()

        query = d['query'].to(device)
        bio_label = d['bio'].to(device)
        domain = d['domains'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)

        print(query)
        print(domain)

        outputs = model(query, attention_mask)
        print(outputs)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(bio_label, targets)
        loss.backward()
        losses.append(loss.item())

        correct_predictions += torch.sum(preds == bio_label)

        optim.step()
        scheduler.step()
    
    return losses