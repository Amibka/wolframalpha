"""
Полная Seq2Seq Transformer модель: текст → SymPy код
"""
import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder


class Seq2SeqTransformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            d_model=256,
            n_heads=8,
            n_encoder_layers=4,
            n_decoder_layers=4,
            d_ff=1024,
            max_len=128,
            dropout=0.1,
            pad_idx=0
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size, d_model, n_heads, n_encoder_layers, d_ff, max_len, dropout
        )

        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_heads, n_decoder_layers, d_ff, max_len, dropout
        )

        self.pad_idx = pad_idx
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def make_src_mask(self, src):
        """Маска для padding в source"""
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_tgt_mask(self, tgt):
        """
        Маска для target:
        1. Padding mask
        2. Causal mask (не смотреть в будущее)
        """
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)

        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()

        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def forward(self, src, tgt):
        """
        Args:
            src: (batch_size, src_len) - входной текст
            tgt: (batch_size, tgt_len) - целевой SymPy код

        Returns:
            output: (batch_size, tgt_len, vocab_size)
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        encoder_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        return output

    def encode(self, src):
        """Кодирует входной текст"""
        src_mask = self.make_src_mask(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask):
        """Декодирует один шаг"""
        tgt_mask = self.make_tgt_mask(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)


def create_model(config, src_vocab_size, tgt_vocab_size):
    """Фабрика для создания модели"""
    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_encoder_layers=config["n_encoder_layers"],
        n_decoder_layers=config["n_decoder_layers"],
        d_ff=config["d_ff"],
        max_len=config["max_seq_length"],
        dropout=config["dropout"],
        pad_idx=0
    )

    # Инициализация весов
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model