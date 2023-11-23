import torch
from tqdm import tqdm

def get_embeddings(model, tokenizer, inputs, device):
    inputs = ["[CLS] " + s for s in inputs]
    encoded_inputs = tokenizer(inputs, return_tensors='pt', padding=True).to(device)

    # forward pass through the model to get embeddings
    with torch.no_grad():
        output = model(**encoded_inputs)

    # extract the CLS embedding
    cls_embeddings = output.last_hidden_state[:, 0, :]
    return cls_embeddings

def train_loop(dataloader, model, model_embed, tokenizer, loss_func, optimizer, device, use_tqdm=True):
    # set in training mode
    model.train()

    total_loss = 0

    iterator = tqdm(dataloader, ncols=100) if use_tqdm else dataloader

    for i, (inputs, targets) in enumerate(iterator):
        inputs = get_embeddings(model_embed, tokenizer, inputs, device)
        targets = targets.to(device)

        # reset gradients
        optimizer.zero_grad()

        outputs = model(inputs)

        # shape target same way as output so loss can be calculated
        targets = targets.view(outputs.shape[0], 1).float()

        # calculate loss
        loss = loss_func(outputs, targets)

        # track total loss
        total_loss += loss.item()

        # backward pass
        loss.backward()

        # update parameters
        optimizer.step()

    # could track overall loss and accuracy and return that
    return total_loss

def val_loop(dataloader, model, model_embed, tokenizer, device, metric, confusion, use_tqdm=True):
    # evaluation mode
    model.eval()

    metric.reset()
    confusion.reset()

    iterator = tqdm(dataloader, ncols=100) if use_tqdm else dataloader

    with torch.no_grad():
        for i, (input, target) in enumerate(iterator):

            inputs = get_embeddings(model_embed, tokenizer, inputs, device)
            targets = targets.to(device)

            output = model(input)

            # AUC metric
            metric.update(output.flatten(), target.flatten())

            # Confusion matrix
            confusion.update(output.sigmoid().flatten(), target.flatten())

    return metric.compute(), confusion.compute()