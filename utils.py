import torch
from tqdm import tqdm

def get_embeddings(model, tokenizer, queries, answers, device):
    encoded_inputs = tokenizer([queries]*11, answers, return_tensors='pt', padding=True, truncation=True).to(device)

    # forward pass through the model to get embeddings
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # Extract embeddings for [CLS] tokens (the first token)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    
    # Concatenate query-answer embeddings along columns
    # features = torch.cat(cls_embeddings, dim=0)

    return cls_embeddings

def train_loop(dataloader, model, model_embed, tokenizer, loss_func, optimizer, device, use_tqdm=True):
    model.train()
    epoch_loss = 0

    iterator = tqdm(dataloader, ncols=100) if use_tqdm else dataloader
    
    for queries, all_answers, targets in iterator:
        print(queries)
        print(all_answers)
        targets = (targets.squeeze(0)).to(device)
        print(targets.shape)
        embeddings = get_embeddings(model_embed, tokenizer, queries, all_answers, device)

        optimizer.zero_grad()
        outputs = model(embeddings)

        # Calculate loss
        loss = loss_func(outputs, targets.float())
        loss.backward()
        optimizer.step()
        
        # Track total loss
        epoch_loss += loss.item()
        
    return epoch_loss

def val_loop(dataloader, model, model_embed, tokenizer, loss_func, device, metric, acc, use_tqdm=True):
    model.eval()
    total_loss = 0
    metric.reset()
    acc.reset()
    # confusion.reset()

    iterator = tqdm(dataloader, ncols=100) if use_tqdm else dataloader

    with torch.no_grad():
        for queries, all_answers, targets in iterator:
            targets = (targets.squeeze(0)).to(device)
            embeddings = get_embeddings(model_embed, tokenizer, queries, all_answers, device)

            outputs = model(embeddings)
            
            # Calculate loss
            loss = loss_func(outputs, targets.float())
            total_loss += loss.item()

            # Update metrics
            metric.update(outputs, targets)
            acc.update(outputs, targets)

            # Confusion matrix
            # confusion.update(outputs.sigmoid().flatten(), targets.flatten())

    return total_loss, metric.compute(), acc.compute() #, confusion.compute()