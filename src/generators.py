import torch.nn as nn


class LSTMGenerator(nn.Module):
    """
    A two-layer LSTM-based binary classifier that processes sequential input data and outputs a probability score.

    Args:
        input_dim (int): Number of expected features in the input sequence.
        hidden_dim (int): Number of features in the hidden state of each LSTM layer.
        num_layers (int): Number of stacked LSTM layers (currently reserved for future extension; not used).
        output_dim (int): Dimension of the final output (currently fixed to 1 for binary classification; not used).
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.dense = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dense(x)
        return self.sigmoid(x).squeeze(1).float()


class TransformerGenerator(nn.Module):
    """
    A Transformer encoder-based generator that processes sequential input data and outputs token probabilities.

    Args:
        input_dim (int): Number of features in each input token.
        d_model (int): Dimension of the embedding space and transformer model.
        nhead (int): Number of attention heads in each transformer layer.
        num_layers (int): Number of stacked TransformerEncoder layers.
        dim_feedforward (int): Dimension of the feedforward network within each encoder layer.
        output_dim (int): Number of output classes or features per token.

    """

    def __init__(
        self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return nn.functional.log_softmax(self.output_proj(x), dim=-1)[:, -1, :]


class GRUGenerator(nn.Module):
    """
    A multi-layer GRU-based generator that processes sequential input data and outputs a final prediction.

    Args:
        input_dim (int): Number of expected features in the input sequence.
        hidden_dim (int): Number of features in the hidden state of the GRU.
        num_layers (int): Number of stacked GRU layers.
        output_dim (int): Dimension of the final output vector.

    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUGenerator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out


class RNNGenerator(nn.Module):
    """
    A multi-layer vanilla RNN-based generator that processes sequential input data and outputs a final prediction.

    Args:
        input_dim (int): Number of expected features in the input sequence.
        hidden_dim (int): Number of features in the hidden state of the RNN.
        num_layers (int): Number of stacked RNN layers.
        output_dim (int): Dimension of the final output vector.

    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNGenerator, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out[:, -1, :])
        return out


class LinearModel(nn.Module):
    """
    A simple feedforward neural network that flattens the input and passes it through two linear layers
    with a ReLU activation in between.

    Args:
        input_dim (int): Number of input features (after flattening) for the first linear layer.
        output_dim (int): Number of output features for the final linear layer.
    """

    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)
