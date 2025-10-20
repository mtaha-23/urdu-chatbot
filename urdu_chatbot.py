import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Text normalization utilities
# -----------------------------
import re

DIACRITICS_PATTERN = re.compile('[\u064B-\u0652\u0670]')  # Harakaat + superscript Alef
CHAR_NORMALIZATION_MAP = {
    'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
    'ى': 'ی', 'ي': 'ی', 'ئ': 'ی',
    'ۀ': 'ہ', 'ؤ': 'و'
}


def normalize_urdu(text: str, remove_diacritics: bool = True) -> str:
    if text is None:
        return ''
    if remove_diacritics:
        text = DIACRITICS_PATTERN.sub('', text)
    for k, v in CHAR_NORMALIZATION_MAP.items():
        text = text.replace(k, v)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -----------------------------
# Tokenizer
# -----------------------------
from collections import Counter


class UrduTokenizer:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0

        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'

        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]

    def build_vocabulary(self, sentences, min_freq=2):
        word_freq = Counter()
        for sentence in sentences:
            words = sentence.split()
            word_freq.update(words)

        vocab = self.special_tokens.copy()
        for word, freq in word_freq.items():
            if freq >= min_freq:
                vocab.append(word)

        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(vocab)

    def tokenize(self, sentence, add_special_tokens=True):
        words = sentence.split()
        tokens = []
        if add_special_tokens:
            tokens.append(self.word_to_idx[self.SOS_TOKEN])
        for word in words:
            tokens.append(self.word_to_idx.get(word, self.word_to_idx[self.UNK_TOKEN]))
        if add_special_tokens:
            tokens.append(self.word_to_idx[self.EOS_TOKEN])
        return tokens

    def detokenize(self, tokens):
        words = []
        for token in tokens:
            if token == self.word_to_idx[self.EOS_TOKEN]:
                break
            if token not in [self.word_to_idx[self.PAD_TOKEN], self.word_to_idx[self.SOS_TOKEN]]:
                words.append(self.idx_to_word.get(token, self.UNK_TOKEN))
        return ' '.join(words)


# -----------------------------
# Transformer building blocks
# -----------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attention_output)
        return output, attention_weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 2,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        d_ff: int = 1024,
        max_length: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_padding_mask(self, seq: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        return (seq != pad_token_id).unsqueeze(1)

    def create_look_ahead_mask(self, size: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        src_embedded = self.embedding(src) * np.sqrt(self.d_model)
        src_embedded = self.pos_encoding(src_embedded.transpose(0, 1)).transpose(0, 1)
        src_embedded = self.dropout(src_embedded)
        encoder_output = src_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        return encoder_output

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt_embedded = self.embedding(tgt) * np.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded.transpose(0, 1)).transpose(0, 1)
        tgt_embedded = self.dropout(tgt_embedded)
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)
        return decoder_output

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        output = self.output_projection(decoder_output)
        return output


# -----------------------------
# Decoding helpers
# -----------------------------


def greedy_decode(model: TransformerModel, src: torch.Tensor, tokenizer: UrduTokenizer, device: torch.device, max_length: int = 50) -> str:
    model.eval()
    with torch.no_grad():
        src_mask = model.create_padding_mask(src, tokenizer.word_to_idx[tokenizer.PAD_TOKEN])
        encoder_output = model.encode(src, src_mask)
        decoder_input = torch.tensor([[tokenizer.word_to_idx[tokenizer.SOS_TOKEN]]], device=device)
        for _ in range(max_length):
            tgt_mask = model.create_look_ahead_mask(decoder_input.size(1))
            decoder_output = model.decode(decoder_input, encoder_output, src_mask, tgt_mask)
            next_token_logits = model.output_projection(decoder_output[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1)
            if next_token.item() == tokenizer.word_to_idx[tokenizer.EOS_TOKEN]:
                break
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
        generated_tokens = decoder_input[0].cpu().tolist()
        generated_text = tokenizer.detokenize(generated_tokens)
        return generated_text


def beam_search_decode(
    model: TransformerModel,
    src: torch.Tensor,
    tokenizer: UrduTokenizer,
    device: torch.device,
    beam_size: int = 3,
    max_length: int = 50,
) -> str:
    model.eval()
    with torch.no_grad():
        src_mask = model.create_padding_mask(src, tokenizer.word_to_idx[tokenizer.PAD_TOKEN])
        encoder_output = model.encode(src, src_mask)
        beams: list[tuple[list[int], float]] = [([tokenizer.word_to_idx[tokenizer.SOS_TOKEN]], 0.0)]
        for _ in range(max_length):
            new_beams: list[tuple[list[int], float]] = []
            for beam_tokens, beam_score in beams:
                if beam_tokens[-1] == tokenizer.word_to_idx[tokenizer.EOS_TOKEN]:
                    new_beams.append((beam_tokens, beam_score))
                    continue
                decoder_input = torch.tensor([beam_tokens], device=device)
                tgt_mask = model.create_look_ahead_mask(decoder_input.size(1))
                decoder_output = model.decode(decoder_input, encoder_output, src_mask, tgt_mask)
                next_token_logits = model.output_projection(decoder_output[:, -1, :])
                next_token_probs = F.log_softmax(next_token_logits, dim=-1)
                top_probs, top_tokens = torch.topk(next_token_probs, beam_size)
                for i in range(beam_size):
                    token = top_tokens[0][i].item()
                    prob = top_probs[0][i].item()
                    new_beam_tokens = beam_tokens + [token]
                    new_beam_score = beam_score + prob
                    new_beams.append((new_beam_tokens, new_beam_score))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            if all(beam[0][-1] == tokenizer.word_to_idx[tokenizer.EOS_TOKEN] for beam in beams):
                break
        best_beam_tokens, _ = beams[0]
        generated_text = tokenizer.detokenize(best_beam_tokens)
        return generated_text


# -----------------------------
# Chatbot wrapper
# -----------------------------


class UrduChatbot:
    def __init__(self, model_path: str, tokenizer: UrduTokenizer, device: torch.device):
        self.device = device
        self.tokenizer = tokenizer
        # PyTorch 2.6 defaults weights_only=True which breaks for older checkpoints containing objects
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            # For older torch versions without weights_only param
            checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {})

        def _build_model_from_config(conf: dict) -> TransformerModel:
            return TransformerModel(
                vocab_size=self.tokenizer.vocab_size,
                d_model=conf.get('d_model', 256),
                num_heads=conf.get('num_heads', 2),
                num_encoder_layers=conf.get('num_encoder_layers', 2),
                num_decoder_layers=conf.get('num_decoder_layers', 2),
                d_ff=conf.get('d_ff', 1024),
                max_length=conf.get('max_length', 50),
                dropout=conf.get('dropout', 0.1),
            )

        state_dict = checkpoint['model_state_dict']

        # First try: use config from checkpoint
        try:
            self.model = _build_model_from_config(config)
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            # Fallback: infer architecture from state_dict shapes
            vocab_size = state_dict['embedding.weight'].shape[0]
            inferred_d_model = state_dict['embedding.weight'].shape[1]
            inferred_max_len = state_dict.get('pos_encoding.pe', torch.empty(50, 1, inferred_d_model)).shape[0]

            # Count encoder/decoder layers
            enc_layer_indices = set()
            dec_layer_indices = set()
            for k in state_dict.keys():
                if k.startswith('encoder_layers.'):
                    try:
                        idx = int(k.split('.')[1])
                        enc_layer_indices.add(idx)
                    except Exception:
                        pass
                elif k.startswith('decoder_layers.'):
                    try:
                        idx = int(k.split('.')[1])
                        dec_layer_indices.add(idx)
                    except Exception:
                        pass
            inferred_num_enc = (max(enc_layer_indices) + 1) if enc_layer_indices else config.get('num_encoder_layers', 2)
            inferred_num_dec = (max(dec_layer_indices) + 1) if dec_layer_indices else config.get('num_decoder_layers', 2)

            # Infer d_ff
            ff_key = 'encoder_layers.0.feed_forward.linear1.weight'
            inferred_d_ff = state_dict.get(ff_key, torch.empty(1024, inferred_d_model)).shape[0]

            # Choose a valid num_heads
            num_heads = config.get('num_heads', 2)
            if inferred_d_model % num_heads != 0:
                # pick a divisor (prefer 4, then 2, else 1)
                for candidate in [8, 4, 2]:
                    if candidate <= inferred_d_model and inferred_d_model % candidate == 0:
                        num_heads = candidate
                        break
                else:
                    num_heads = 1

            self.model = TransformerModel(
                vocab_size=vocab_size,
                d_model=inferred_d_model,
                num_heads=num_heads,
                num_encoder_layers=inferred_num_enc,
                num_decoder_layers=inferred_num_dec,
                d_ff=inferred_d_ff,
                max_length=inferred_max_len,
                dropout=config.get('dropout', 0.1),
            )
            self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    def preprocess_input(self, text: str, max_length: int | None = None) -> torch.Tensor:
        normalized_text = normalize_urdu(text)
        tokens = self.tokenizer.tokenize(normalized_text, add_special_tokens=False)
        max_len = max_length or getattr(self.model, 'max_length', 50)
        while len(tokens) < max_len:
            tokens.append(self.tokenizer.word_to_idx[self.tokenizer.PAD_TOKEN])
        tokens = tokens[:max_len]
        return torch.tensor([tokens], device=self.device)

    def generate_response(self, input_text: str, method: str = 'greedy', beam_size: int = 3) -> str:
        src = self.preprocess_input(input_text)
        if method == 'greedy':
            return greedy_decode(self.model, src, self.tokenizer, self.device, max_length=self.model.max_length)
        if method == 'beam':
            return beam_search_decode(self.model, src, self.tokenizer, self.device, beam_size=beam_size, max_length=self.model.max_length)
        raise ValueError("Method must be 'greedy' or 'beam'")

    def chat(self, input_text: str, method: str = 'greedy') -> str:
        try:
            return self.generate_response(input_text, method)
        except Exception as e:
            return f"Error generating response: {str(e)}"


__all__ = [
    'normalize_urdu',
    'UrduTokenizer',
    'TransformerModel',
    'UrduChatbot',
    'greedy_decode',
    'beam_search_decode',
]


