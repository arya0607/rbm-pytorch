import random
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
            v, v_gibbs, _ = model(data.view(-1, 784), k)
            loss = model.free_energy(v) - model.free_energy(v_gibbs)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()
        epoch_loss.append(np.mean(loss_))
        print('Epoch %d\t Loss=%.4f' % (epoch, np.mean(loss_)))
    
    return epoch_loss

def test_for_inpaint(model, test_loader, device, inpainting_technique = 'random_top_half', k = 1000, plot = False):
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
    print('Inpainting Technique:', inpainting_technique, '; k:', k)
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
        crop_size = 10
        if inpainting_technique == 'random_top_half':
            v_input[:, :length // 2] = torch.rand((batch_size, length // 2)).to(device)
        elif inpainting_technique == 'zero_top_half':
            v_input[:, :length // 2] = 0
        elif inpainting_technique == 'zero_center_crop':
            v_input = v_input.reshape(-1, 28, 28)
            start = (28 - crop_size) // 2
            end = start + crop_size
            v_input[:, start:end:, start:end] = 0
            v_input = v_input.reshape(-1, 28*28)

    
        if inpainting_technique != 'zero_center_crop':
            v_mask = torch.ones((batch_size, length)).to(device)
            v_mask[:, :length // 2 ] = 0
        if inpainting_technique == 'zero_center_crop':
            start = (28 - crop_size) // 2
            end = start + crop_size
            v_mask = torch.ones((batch_size, length)).to(device)
            v_mask = v_mask.reshape(-1, 28, 28)
            v_mask[:, start:end:, start:end] = 0
            v_mask = v_mask.reshape(-1, 28*28)
        _,v_gibbs, _ = model(v_input, v_true=v_true, v_mask=v_mask, k=k,device=device)
        # compute accuracy
        mask_zeros = v_mask == 0
        correct += torch.sum(v_gibbs[mask_zeros] == v_true[mask_zeros]).item()
        total += torch.sum(mask_zeros).item()
        if plot:
            print(correct, total)
            plot_examples = 10
            total_examples = v_input.size(0)
            random_indices = [random.randint(0, total_examples - 1) for _ in range(plot_examples)]
            v_input = v_input[random_indices]
            v_true = v_true[random_indices]
            v_mask = v_mask[random_indices]
            print('k: ',k)
            _,v_gibbs, intermediate = model(v_input, v_true=v_true, v_mask=v_mask, k=k,device=device, log_every=k // 10)
            print('intermediate length ', len(intermediate))
            fig, axs = plt.subplots(plot_examples, len(intermediate) + 1, figsize=(len(intermediate) + 1, plot_examples))
            for it in range(plot_examples):
                for j in range(len(intermediate)):
                    pred = intermediate[j][it]
                    pred = pred.view(28, 28)
                    axs[it, j].imshow(pred.cpu().detach().numpy(), cmap='gray')
                # pred = v_gibbs[it]
                # pred = pred.view(28, 28)
                # axs[i, 0].imshow(pred.cpu().detach().numpy(), cmap='gray')

                # input_img = v_input[it]
                # input_img = input_img.view(28, 28)
                # axs[i, 1].imshow(input_img.cpu().detach().numpy(), cmap='gray')

                img = v_true[it]
                img = img.view(28, 28)
                axs[it, -1].imshow(img.cpu().detach().numpy(), cmap='gray')

            axs[0, -1].set_xlabel('True Image')
            plt.show()
            plt.close()
            break
        # print('Batch Size:', batch_size, 'Accuracy: %.2f%%' % (accuracy * 100))

    accuracy = correct / total
    print('correct:', correct, 'total:', total)
    print('Final Accuracy: %.2f%%' % (accuracy * 100))
    return accuracy


