import torch
from tqdm import tqdm

def train_loop(dataloader, model, loss_func, optimizer, device, use_tqdm=True):
    # set in training mode
    model.train()

    total_loss = 0

    iterator = tqdm(dataloader, ncols=100) if use_tqdm else dataloader

    for i, (input, target) in enumerate(iterator):

        input, target = input.to(device), target.to(device)

        # reset gradients
        optimizer.zero_grad()

        # get model prediction - need to specify logits for inception and ViT since it returns a dict
        if logits:
            output = model(input).logits
        else:
            output = model(input)

        # shape target same way as output so loss can be calculated
        target = target.view(output.shape[0], 1).float()

        # calculate loss
        loss = loss_func(output, target)

        # track total loss
        total_loss += loss.item()

        # backward pass
        loss.backward()

        # update parameters
        optimizer.step()

    # could track overall loss and accuracy and return that
    return total_loss