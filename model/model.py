import torch
import torch.nn as nn
from model.decoder import Decoder
from model.htr_decoder import HTRDecoder


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.htr = getattr(args, 'htr', False)
        if self.htr:
            self.decoder = HTRDecoder(args).to(self.device)
        else:
            self.decoder = Decoder(args).to(self.device)

    def forward(self, x, max_retrievals=None):
        decoder_output = self.decoder(x, max_retrievals)
        if self.htr:
            cost, ll, target_logp = decoder_output
            return cost, ll, target_logp
        else:
            cost, ll = decoder_output
            return cost, ll
