import torch
import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers):
        super(EncoderLSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, output_size, dropout_p=0.3):
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers,
                            dropout=dropout_p if num_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size, max_target_length, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size
        self.max_target_length = max_target_length
        self.device = device


class Decoder:
    def __init__(self, model, source_vocab, target_vocab, source_idx2char, target_idx2char, device):
        self.model = model
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.source_idx2char = source_idx2char
        self.target_idx2char = target_idx2char
        self.device = device
        self.sos_token = target_vocab['<sos>']
        self.eos_token = target_vocab['<eos>']

    def greedy_decode(self, source_sequence, max_length=50):
        self.model.eval()
        with torch.no_grad():
            if len(source_sequence.shape) == 1:
                source_sequence = source_sequence.unsqueeze(0)
            hidden, cell = self.model.encoder(source_sequence)
            decoded = [self.sos_token]
            decoder_input = torch.tensor([[self.sos_token]], device=self.device)
            for _ in range(max_length):
                output, hidden, cell = self.model.decoder(decoder_input, hidden, cell)
                token = output.argmax(dim=1).item()
                decoded.append(token)
                if token == self.eos_token:
                    break
                decoder_input = torch.tensor([[token]], device=self.device)
        return decoded

    def beam_search_decode(self, source_sequence, beam_width=5, max_length=50):
        self.model.eval()
        with torch.no_grad():
            if len(source_sequence.shape) == 1:
                source_sequence = source_sequence.unsqueeze(0)
            hidden, cell = self.model.encoder(source_sequence)
            beams = [(0.0, [self.sos_token], hidden, cell)]
            completed = []
            for _ in range(max_length):
                candidates = []
                for log_prob, seq, h, c in beams:
                    if seq[-1] == self.eos_token:
                        completed.append((log_prob, seq))
                        continue
                    inp = torch.tensor([[seq[-1]]], device=self.device)
                    output, h_new, c_new = self.model.decoder(inp, h, c)
                    log_probs = torch.log_softmax(output, dim=1)
                    top_log_probs, top_idx = torch.topk(log_probs, beam_width, dim=1)
                    for i in range(beam_width):
                        token_id = top_idx[0, i].item()
                        new_prob = log_prob + top_log_probs[0, i].item()
                        candidates.append((new_prob, seq + [token_id], h_new, c_new))
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_width]
                if all(s[-1] == self.eos_token for _, s, _, _ in beams):
                    completed.extend([(p, s) for p, s, _, _ in beams])
                    break
            if completed:
                return max(completed, key=lambda x: x[0])[1]
            return [self.sos_token, self.eos_token]
