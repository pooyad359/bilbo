import torch
from torch import nn


class BilboProjection(nn.Module):
    def __init__(self, input_size=8, hidden_size=128):
        super(BilboProjection, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return self.projection(x)


# class BilboLayer(nn.Module):
#     def __init__(self, hidden_size, activation=nn.ReLU()):
#         super(BilboLayer, self).__init__()

#         self.hidden_size = hidden_size
#         self.operator = nn.Linear(hidden_size * 2, hidden_size)
#         # self.layer_norm = nn.LayerNorm(hidden_size)
#         self.activation = activation

#     def forward(self, x: torch.Tensor):
#         batch_size, seq_length, hidden_size = x.shape
#         x1 = x.unsqueeze(1).repeat(1, seq_length, 1, 1)
#         x2 = x.unsqueeze(2).repeat(1, 1, seq_length, 1)
#         x_in = torch.cat([x1, x2], dim=-1).reshape(batch_size, -1, hidden_size * 2)
#         out = self.operator(x_in).reshape(batch_size, seq_length, seq_length, hidden_size)
#         # out = self.layer_norm(out)
#         out = out.mean(dim=2)
#         return self.activation(out)


# class BilboOrderHead(nn.Module):
#     def __init__(self, hidden_size):
#         super(BilboOrderHead, self).__init__()

#         self.hidden_size = hidden_size
#         self.operator = nn.Linear(hidden_size * 2, hidden_size)
#         # self.layer_norm = nn.LayerNorm(hidden_size)
#         self.head = nn.Linear(hidden_size, 1)

#     def forward(self, x: torch.Tensor):
#         batch_size, seq_length, hidden_size = x.shape
#         x1 = x.unsqueeze(1).repeat(1, seq_length, 1, 1)
#         x2 = x.unsqueeze(2).repeat(1, 1, seq_length, 1)
#         x_in = torch.cat([x1, x2], dim=-1).reshape(batch_size, -1, hidden_size * 2)
#         out = self.operator(x_in)  # .reshape(batch_size, seq_length, seq_length, hidden_size)
#         out = self.head(out).reshape(batch_size, seq_length, seq_length)
#         # out = self.layer_norm(out)
#         return out


# class BilboModel(nn.Module):
#     def __init__(self, input_size=8, hidden_size=128, n_layers=1):
#         super(BilboModel, self).__init__()

#         self.hidden_size = hidden_size
#         self.input_size = input_size
#         self.n_layers = n_layers
#         self.projection = BilboProjection(input_size=input_size, hidden_size=hidden_size)
#         self.layers = [BilboLayer(hidden_size) for _ in range(n_layers)]
#         self.head = BilboOrderHead(hidden_size)
#         self.model = nn.Sequential(self.projection, *self.layers, self.head)


#     def forward(self, x):
#         return self.model(x)


class BilboAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BilboAttention, self).__init__()

        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def attention(self, x):
        q = self.query(x)
        k = self.key(x)
        return self.softmax(q @ k.transpose(-2, -1) / self.hidden_size**0.5)

    def forward(self, x):
        return self.attention(x) @ self.value(x)


class BilboHead(nn.Module):
    def __init__(self, hidden_size):
        super(BilboHead, self).__init__()

        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        q = self.query(x)
        k = self.key(x)
        return q @ k.transpose(-2, -1) / self.hidden_size**0.5


class BilboTransformerLayer(nn.Module):
    def __init__(self, hidden_size):
        super(BilboTransformerLayer, self).__init__()

        self.hidden_size = hidden_size
        self.attention = BilboAttention(hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.operator = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x + self.attention(x)
        x = self.layer_norm1(x)
        x = x + self.operator(x)
        x = self.layer_norm2(x)
        return self.activation(x)


class BilboModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, n_layers=1):
        super(BilboModel, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.projection = BilboProjection(input_size=input_size, hidden_size=hidden_size)
        self.layers = [BilboTransformerLayer(hidden_size) for _ in range(n_layers)]
        self.head = BilboHead(hidden_size)
        self.model = nn.Sequential(self.projection, *self.layers, self.head)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        y = self.model(x)
        return y

    def loss(self, x, y):
        yhat = self.forward(x)
        # pos_weight = torch.ones(y.shape[1:])
        _loss = self.bce(yhat, y)
        return _loss, yhat
