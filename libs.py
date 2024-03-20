import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch
from tqdm import tqdm


def show_and_save(img, file_name):
    r"""Show and save the image.

    Args:
        img (Tensor): The image.
        file_name (Str): The destination.

    """
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    plt.imshow(npimg, cmap='gray')
    plt.imsave(f, npimg)


def train(model, train_loader, device, n_epochs=20, lr=0.001,k=None):
    r"""Train a RBM model.

    Args:
        model: The model.
        train_loader (DataLoader): The data loader.
        n_epochs (int, optional): The number of epochs. Defaults to 20.
        lr (Float, optional): The learning rate. Defaults to 0.01.

    Returns:
        The trained model.

    """
    # optimizer
    train_op = optim.Adam(model.parameters(), lr)

    # train the RBM model
    model.train()
    epoch_loss = []

    for epoch in tqdm(range(n_epochs)):
        loss_ = []
        for _, (data, target) in enumerate(train_loader):
            data = data.to(device)
            v, v_gibbs = model(data.view(-1, 784), k)
            loss = model.free_energy(v) - model.free_energy(v_gibbs)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()
        epoch_loss.append(np.mean(loss_))
        print('Epoch %d\t Loss=%.4f' % (epoch, np.mean(loss_)))
    
    return epoch_loss

def test_for_inpaint(model, test_loader, device, random_values = True, k = 1000):
    r"""Test the model for inpainting.
    Top half of the image is masked and the model is asked to reconstruct it.

    Args:
        model: The model.
        test_loader (DataLoader): The data loader.
        k (int, optional): The number of Gibbs sampling. Defaults to 1000.
        random_values (bool, optional): Randomly initialize the visible variable. Defaults to True. If False, the true visible variables are set to 0.
    """
    # test the RB
    print('Testing the RBM model for inpainting...')
    print('Random Values:', random_values, '; k:', k)
    model.eval()
    correct = 0
    total = 0
    v_mask = None
    for i, (data, target) in tqdm(enumerate(test_loader)):
        data = data.to(device)
        v_true= data.view(-1, 784)
        v_input = v_true.clone()
        # print(data.device, v_true.device, v_input.device, next(model.parameters()).device)
        batch_size,length = v_input.size()
        if random_values:
            v_input[:, :length // 2].bernoulli()
        else:
            v_input[:, :length // 2] = 0
        if v_mask is None or not batch_size == v_mask.size(0):
            v_mask = torch.ones((batch_size, length)).to(device)
            v_mask[:, :length // 2 ] = 0
        _,v_gibbs = model(v_input, v_true=v_true, v_mask=v_mask, k=k)
        # compute accuracy
        v_pred = v_gibbs[:, :length // 2]
        accuracy = torch.eq(v_pred, v_true[:, :length // 2]).sum().item() / (batch_size * (length // 2))
        correct += torch.eq(v_pred, v_true[:, :length // 2]).sum().item()
        total += batch_size * (length // 2)
        # print('Batch Size:', batch_size, 'Accuracy: %.2f%%' % (accuracy * 100))
        
    accuracy = correct / total
    print('correct:', correct, 'total:', total)
    print('Final Accuracy: %.2f%%' % (accuracy * 100))
    return accuracy
        