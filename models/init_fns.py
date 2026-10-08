
import torch.nn as nn


def init_xavier(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def init_kaiming(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.zero_()


def init_orthogonal(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

INIT_FN_MAP = {
    "xavier_normal":   nn.init.xavier_normal_,
    # "xavier_uniform":  nn.init.xavier_uniform_,
    # "kaiming_normal":  nn.init.kaiming_normal_,
    # "kaiming_uniform": nn.init.kaiming_uniform_,
    # "orthogonal":      nn.init.orthogonal_,
}
