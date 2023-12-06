import torch
from tqdm import tqdm
from sklearn.decomposition import PCA

def get_embeddings(model, tokenizer, queries, answer_list, device, reduce_dim=False):
    combined_inputs = []
    
    for qid, query in enumerate(queries):
        combined_strings = []
        for answers in answer_list:
            combined_strings.append("[CLS] " + query + " [SEP] " + answers[qid])
        combined_inputs.extend(combined_strings)

    encoded_inputs = tokenizer(combined_inputs, return_tensors='pt', padding=True, truncation=True).to(device)
    
    # forward pass through the model to get embeddings
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # Extract embeddings for [CLS] tokens (the first token)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]

    if reduce_dim:
        embeddings_array = cls_embeddings.cpu().numpy()
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings_array)
        cls_embeddings = torch.tensor(reduced_embeddings, device=device)

    return cls_embeddings

def train_loop(dataloader, model, model_embed, tokenizer, loss_func, optimizer, device, use_tqdm=True, reduce=False):
    model.train()
    epoch_loss = 0

    iterator = tqdm(dataloader, ncols=100) if use_tqdm else dataloader
    
    for queries, all_answers, targets in iterator:
        targets = targets.squeeze(0).to(device)
        embeddings = get_embeddings(model_embed, tokenizer, queries, all_answers, device, reduce)

        optimizer.zero_grad()
        outputs = model(embeddings)

        # Calculate loss
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Track total loss
        epoch_loss += loss.item() / len(all_answers)
        
    return epoch_loss

def val_loop(dataloader, model, model_embed, tokenizer, loss_func, device, metric, acc, use_tqdm=True, reduce=False):
    model.eval()
    total_loss = 0
    metric.reset()
    acc.reset()
    # confusion.reset()

    iterator = tqdm(dataloader, ncols=100) if use_tqdm else dataloader

    with torch.no_grad():
        for queries, all_answers, targets in iterator:
            targets = targets.squeeze(0).to(device)
            embeddings = get_embeddings(model_embed, tokenizer, queries, all_answers, device, reduce)

            outputs = model(embeddings)
            
            # Calculate loss
            loss = loss_func(outputs, targets)
            total_loss += loss.item() / len(all_answers)

            # Update metrics
            metric.update(outputs.sigmoid(), targets)
            acc.update(outputs.sigmoid(), targets)

            # Confusion matrix
            # confusion.update(outputs.sigmoid().flatten(), targets.flatten())

    return total_loss, metric.compute(), acc.compute() #, confusion.compute()