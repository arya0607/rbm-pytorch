import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class RBM(nn.Module):
    r"""Restricted Boltzmann Machine.

    Args:
        n_vis (int, optional): The size of visible layer. Defaults to 784.
        n_hid (int, optional): The size of hidden layer. Defaults to 128.
        k (int, optional): The number of Gibbs sampling. Defaults to 1.
    """

    def __init__(self, n_vis=784, n_hid=128, k=1):
        """Create a RBM."""
        super(RBM, self).__init__()
        self.v = nn.Parameter(torch.randn(1, n_vis))
        self.h = nn.Parameter(torch.randn(1, n_hid))
        self.W = nn.Parameter(torch.randn(n_hid, n_vis))
        self.k = k

    def visible_to_hidden(self, v):
        r"""Conditional sampling a hidden variable given a visible variable.

        Args:
            v (Tensor): The visible variable.

        Returns:
            Tensor: The hidden variable.

        """
        p = torch.sigmoid(F.linear(v, self.W, self.h))
        return p.bernoulli()

    def hidden_to_visible(self, h):
        r"""Conditional sampling a visible variable given a hidden variable.

        Args:
            h (Tendor): The hidden variable.

        Returns:
            Tensor: The visible variable.

        """
        p = torch.sigmoid(F.linear(h, self.W.t(), self.v))
        return p.bernoulli()

    def free_energy(self, v):
        r"""Free energy function.

        .. math::
            \begin{align}
                F(x) &= -\log \sum_h \exp (-E(x, h)) \\
                &= -a^\top x - \sum_j \log (1 + \exp(W^{\top}_jx + b_j))\,.
            \end{align}

        Args:
            v (Tensor): The visible variable.

        Returns:
            FloatTensor: The free energy value.

        """
        v_term = torch.matmul(v, self.v.t())
        w_x_h = F.linear(v, self.W, self.h)
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return torch.mean(-h_term - v_term)

    def forward(self, v, v_mask = None, v_true = None, k = None, device = 'cpu', log_every = None):
        r"""Compute the real and generated examples.

        Args:
            v (Tensor): The visible variable.
            k is not None when we are generating the examples

        Returns:
            (Tensor, Tensor): The real and generagted variables.

        """
        if k is None:
            # this is the case when we are training the model
            k = self.k
        h = self.visible_to_hidden(v)
        intermediate = []
        if log_every is not None:
            intermediate.append(v)
        for i in range(k):
            v_gibb = self.hidden_to_visible(h)
            if v_mask is not None and v_true is not None:
                v_gibb = torch.where(v_mask == 1, v_true, v_gibb)
            h = self.visible_to_hidden(v_gibb)
            if not device == 'cpu':
                torch.cuda.empty_cache()
            if log_every is not None and (i + 1) % log_every == 0:
                intermediate.append(v_gibb)
            
        return v, v_gibb, intermediate
