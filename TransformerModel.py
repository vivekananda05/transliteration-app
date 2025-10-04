# model.py
import torch
import torch.nn as nn
import math

# ------------------------
# Positional Encoding
# ------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# ------------------------
# Transformer Model
# ------------------------
class TransformerTransliterator(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model=256, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=512, dropout=0.1):
        super(TransformerTransliterator, self).__init__()
        self.d_model = d_model
        self.source_embedding = nn.Embedding(source_vocab_size, d_model)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=False
        )
        self.output_projection = nn.Linear(d_model, target_vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.source_embedding.weight.data.uniform_(-initrange, initrange)
        self.target_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq, pad_idx=0):
        return (seq == pad_idx)

    def forward(self, source, target):
        source = source.transpose(0, 1)  # (seq_len, batch)
        target = target.transpose(0, 1)  # (seq_len, batch)

        seq_len = target.size(0)
        target_mask = self.generate_square_subsequent_mask(seq_len).to(target.device)
        src_padding_mask = self.create_padding_mask(source.transpose(0, 1)).to(source.device)
        tgt_padding_mask = self.create_padding_mask(target.transpose(0, 1)).to(target.device)

        src_emb = self.pos_encoding(self.source_embedding(source) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.target_embedding(target) * math.sqrt(self.d_model))

        output = self.transformer(src_emb, tgt_emb,
                                  tgt_mask=target_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask)
        output = self.output_projection(output)
        return output.transpose(0, 1)  # (batch, seq_len, vocab_size)


# ------------------------
# Encode / Decode utils
# ------------------------
def encode_sequence(sequence, vocab, max_length):
    encoded = [vocab['<sos>']]
    for char in sequence:
        encoded.append(vocab.get(char, vocab['<unk>']))
    encoded.append(vocab['<eos>'])
    if len(encoded) < max_length:
        encoded.extend([vocab['<pad>']] * (max_length - len(encoded)))
    else:
        encoded = encoded[:max_length-1] + [vocab['<eos>']]
    return encoded

def decode_sequence(sequence, idx2char):
    decoded = []
    for idx in sequence:
        if idx in [0, 2]:  # pad or eos
            break
        if idx in idx2char and idx2char[idx] not in ['<sos>', '<unk>']:
            decoded.append(idx2char[idx])
    return ''.join(decoded)


# ------------------------
# Greedy Decoding
# ------------------------
def transliterate_greedy(model, word, source_vocab, target_vocab,
                         source_idx2char, target_idx2char,
                         max_length=50, device="cpu"):
    model.eval()
    with torch.no_grad():
        src_encoded = encode_sequence(word, source_vocab, max_length)
        src_tensor = torch.tensor([src_encoded]).to(device)
        tgt_tensor = torch.tensor([[target_vocab['<sos>']]]).to(device)

        decoded_tokens = []
        for _ in range(max_length):
            output = model(src_tensor, tgt_tensor)
            next_token = output[:, -1].argmax(dim=-1).item()
            if next_token == target_vocab['<eos>']:
                break
            decoded_tokens.append(next_token)
            tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[next_token]]).to(device)], dim=1)

    return decode_sequence(decoded_tokens, target_idx2char)


# ------------------------
# Beam Search Decoder
# ------------------------
class TransformerBeamSearchDecoder:
    def __init__(self, model, source_vocab, target_vocab,
                 source_idx2char, target_idx2char, device):
        self.model = model
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.source_idx2char = source_idx2char
        self.target_idx2char = target_idx2char
        self.device = device

    def beam_search_decode(self, word, beam_width=5, max_length=50):
        self.model.eval()
        with torch.no_grad():
            src_encoded = encode_sequence(word, self.source_vocab, max_length)
            src_tensor = torch.tensor([src_encoded]).to(self.device)

            beams = [(0.0, [self.target_vocab['<sos>']])]
            completed = []

            for _ in range(max_length):
                new_beams = []
                for score, seq in beams:
                    tgt_tensor = torch.tensor([seq]).to(self.device)
                    output = self.model(src_tensor, tgt_tensor)
                    probs = torch.log_softmax(output[:, -1], dim=-1)

                    topk = torch.topk(probs, beam_width)
                    for i in range(beam_width):
                        next_token = topk.indices[0, i].item()
                        next_score = score + topk.values[0, i].item()
                        new_seq = seq + [next_token]
                        if next_token == self.target_vocab['<eos>']:
                            completed.append((next_score, new_seq))
                        else:
                            new_beams.append((next_score, new_seq))

                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
                if not beams:
                    break

            if completed:
                best_seq = max(completed, key=lambda x: x[0])[1]
            else:
                best_seq = beams[0][1]

        return decode_sequence(best_seq[1:], self.target_idx2char)
# model.py
import torch
import torch.nn as nn
import math

# ------------------------
# Positional Encoding
# ------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# ------------------------
# Transformer Model
# ------------------------
class TransformerTransliterator(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model=256, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=512, dropout=0.1):
        super(TransformerTransliterator, self).__init__()
        self.d_model = d_model
        self.source_embedding = nn.Embedding(source_vocab_size, d_model)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=False
        )
        self.output_projection = nn.Linear(d_model, target_vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.source_embedding.weight.data.uniform_(-initrange, initrange)
        self.target_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq, pad_idx=0):
        return (seq == pad_idx)

    def forward(self, source, target):
        source = source.transpose(0, 1)  # (seq_len, batch)
        target = target.transpose(0, 1)  # (seq_len, batch)

        seq_len = target.size(0)
        target_mask = self.generate_square_subsequent_mask(seq_len).to(target.device)
        src_padding_mask = self.create_padding_mask(source.transpose(0, 1)).to(source.device)
        tgt_padding_mask = self.create_padding_mask(target.transpose(0, 1)).to(target.device)

        src_emb = self.pos_encoding(self.source_embedding(source) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.target_embedding(target) * math.sqrt(self.d_model))

        output = self.transformer(src_emb, tgt_emb,
                                  tgt_mask=target_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask)
        output = self.output_projection(output)
        return output.transpose(0, 1)  # (batch, seq_len, vocab_size)


# ------------------------
# Encode / Decode utils
# ------------------------
def encode_sequence(sequence, vocab, max_length):
    encoded = [vocab['<sos>']]
    for char in sequence:
        encoded.append(vocab.get(char, vocab['<unk>']))
    encoded.append(vocab['<eos>'])
    if len(encoded) < max_length:
        encoded.extend([vocab['<pad>']] * (max_length - len(encoded)))
    else:
        encoded = encoded[:max_length-1] + [vocab['<eos>']]
    return encoded

def decode_sequence(sequence, idx2char):
    decoded = []
    for idx in sequence:
        if idx in [0, 2]:  # pad or eos
            break
        if idx in idx2char and idx2char[idx] not in ['<sos>', '<unk>']:
            decoded.append(idx2char[idx])
    return ''.join(decoded)


# ------------------------
# Greedy Decoding
# ------------------------
def transliterate_greedy(model, word, source_vocab, target_vocab,
                         source_idx2char, target_idx2char,
                         max_length=50, device="cpu"):
    model.eval()
    with torch.no_grad():
        src_encoded = encode_sequence(word, source_vocab, max_length)
        src_tensor = torch.tensor([src_encoded]).to(device)
        tgt_tensor = torch.tensor([[target_vocab['<sos>']]]).to(device)

        decoded_tokens = []
        for _ in range(max_length):
            output = model(src_tensor, tgt_tensor)
            next_token = output[:, -1].argmax(dim=-1).item()
            if next_token == target_vocab['<eos>']:
                break
            decoded_tokens.append(next_token)
            tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[next_token]]).to(device)], dim=1)

    return decode_sequence(decoded_tokens, target_idx2char)


# ------------------------
# Beam Search Decoder
# ------------------------
class TransformerBeamSearchDecoder:
    def __init__(self, model, source_vocab, target_vocab,
                 source_idx2char, target_idx2char, device):
        self.model = model
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.source_idx2char = source_idx2char
        self.target_idx2char = target_idx2char
        self.device = device

    def beam_search_decode(self, word, beam_width=5, max_length=50):
        self.model.eval()
        with torch.no_grad():
            src_encoded = encode_sequence(word, self.source_vocab, max_length)
            src_tensor = torch.tensor([src_encoded]).to(self.device)

            beams = [(0.0, [self.target_vocab['<sos>']])]
            completed = []

            for _ in range(max_length):
                new_beams = []
                for score, seq in beams:
                    tgt_tensor = torch.tensor([seq]).to(self.device)
                    output = self.model(src_tensor, tgt_tensor)
                    probs = torch.log_softmax(output[:, -1], dim=-1)

                    topk = torch.topk(probs, beam_width)
                    for i in range(beam_width):
                        next_token = topk.indices[0, i].item()
                        next_score = score + topk.values[0, i].item()
                        new_seq = seq + [next_token]
                        if next_token == self.target_vocab['<eos>']:
                            completed.append((next_score, new_seq))
                        else:
                            new_beams.append((next_score, new_seq))

                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
                if not beams:
                    break

            if completed:
                best_seq = max(completed, key=lambda x: x[0])[1]
            else:
                best_seq = beams[0][1]

        return decode_sequence(best_seq[1:], self.target_idx2char)
