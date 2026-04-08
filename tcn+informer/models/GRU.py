import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.input_size = configs.enc_in
        self.output_size = configs.c_out
        self.pred_len = configs.pred_len
        self.dropout = configs.dropout

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        self.fc = nn.Linear(self.hidden_size, self.pred_len)
        self.projection = nn.Linear(1, self.output_size)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch_size = x_enc.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x_enc.device)

        out, _ = self.gru(x_enc, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.unsqueeze(-1)
        out = self.projection(out)

        return out
