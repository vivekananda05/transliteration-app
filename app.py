import streamlit as st
st.set_page_config(page_title="Unified Transliteration App", layout="wide")
import torch, time, random, os
from LstmModel import EncoderLSTM, DecoderLSTM, Seq2Seq, Decoder
from TransformerModel import (
    TransformerTransliterator,
    encode_sequence,
    decode_sequence,
    transliterate_greedy,
    TransformerBeamSearchDecoder
)

# Load external CSS
def local_css(file_name="styles.css"):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles.css")

# ============================
# Utility Functions (LSTM)
# ============================
def encode_sequence_lstm(sequence, vocab, max_length):
    encoded = [vocab['<sos>']]
    for ch in sequence:
        encoded.append(vocab.get(ch, vocab['<unk>']))
    encoded.append(vocab['<eos>'])
    if len(encoded) < max_length:
        encoded.extend([vocab['<pad>']] * (max_length - len(encoded)))
    else:
        encoded = encoded[:max_length-1] + [vocab['<eos>']]
    return encoded

def decode_sequence_lstm(sequence, idx2char):
    decoded = []
    for idx in sequence:
        if idx in [0, 2]:  # <pad>, <eos>
            break
        if idx in idx2char:
            decoded.append(idx2char[idx])
    return ''.join(decoded)

@st.cache_resource
def load_lstm_model(path="lstm_transliteration_model.pth", device="cpu"):
    torch.serialization.add_safe_globals([Seq2Seq, EncoderLSTM, DecoderLSTM])
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    source_vocab, target_vocab = checkpoint['source_vocab'], checkpoint['target_vocab']
    source_idx2char, target_idx2char = checkpoint['source_idx2char'], checkpoint['target_idx2char']
    max_source_length, max_target_length = checkpoint['max_source_length'], checkpoint['max_target_length']

    encoder = EncoderLSTM(len(source_vocab), 128, 256, 2)
    decoder = DecoderLSTM(len(target_vocab), 128, 256, 2, len(target_vocab), 0.3)
    model = Seq2Seq(encoder, decoder, len(target_vocab), max_target_length, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    decoder_obj = Decoder(model, source_vocab, target_vocab, source_idx2char, target_idx2char, device)
    return model, decoder_obj, source_vocab, target_vocab, source_idx2char, target_idx2char, max_source_length

# ============================
# Transformer Loader
# ============================
@st.cache_resource
def load_transformer_model(checkpoint_path="best_transformer_model.pth", device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TransformerTransliterator(
        source_vocab_size=len(checkpoint['source_vocab']),
        target_vocab_size=len(checkpoint['target_vocab']),
        d_model=256, nhead=8,
        num_encoder_layers=2, num_decoder_layers=2,
        dim_feedforward=512, dropout=0.1
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    decoder = TransformerBeamSearchDecoder(
        model,
        checkpoint['source_vocab'],
        checkpoint['target_vocab'],
        checkpoint['source_idx2char'],
        checkpoint['target_idx2char'],
        device
    )
    return (
        model,
        checkpoint['source_vocab'],
        checkpoint['target_vocab'],
        checkpoint['source_idx2char'],
        checkpoint['target_idx2char'],
        checkpoint['max_source_length'],
        checkpoint['max_target_length'],
        decoder,
    )

# ============================
# Gemini API Setup
# ============================

from google import genai
try:
    from google import genai
    GEMINI_CLIENT_TYPE = "google.genai.client"
except Exception:
    # fallback older package name
    try:
        import google.generativeai as genai
        GEMINI_CLIENT_TYPE = "google.generativeai"
    except Exception:
        genai = None
        GEMINI_CLIENT_TYPE = None

if genai is None:
    raise RuntimeError("Gemini client library not found. Install the official Google Gen AI SDK (see docs).")

import os
os.environ["GEMINI_API_KEY"] = "AIzaSyDylPDwt3Hi45YEKl9ytjP6ysPIaB5ezTQ"

try:
    client = genai.Client()
except Exception:
    # Some older wrappers use configure
    try:
        genai.configure(api_key=None)  # rely on env var
        client = genai
    except Exception as e:
        raise RuntimeError("Failed to initialize Gemini client. See docs and set GEMINI_API_KEY.") from e
    

GEMINI_MODEL = "gemini-2.5-flash"
from google.genai import types
custom_config = types.GenerateContentConfig(
    temperature=1.0,
    top_p=0.8
)

# ---------------------------
# Transliteration function
# ---------------------------
def gemini_transliterate(word, retries=3):
    """Transliterate a single English word into Hindi (Devanagari)."""
    prompt = (
        f"Transliterate the following English word into Hindi (Devanagari) script.\n\n"
        f"Word: {word}\n\n"
        f"Return only the transliteration inside <output>...</output> tags."
    )

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=custom_config
            )
            text = getattr(response, "text", None)
            if not text and hasattr(response, "candidates"):
                text = response.candidates[0].content.parts[0].text

            if text and "<output>" in text and "</output>" in text:
                return text.split("<output>")[1].split("</output>")[0].strip()
            elif text:
                return text.strip().splitlines()[0].strip()

        except Exception as e:
            wait = (2 ** attempt) + random.random()
            time.sleep(wait)

    return "(Error: Could not transliterate)"


# ============================
# Streamlit UI
# ============================


st.markdown("<h1> English → Hindi Transliteration</h1>", unsafe_allow_html=True)


col_about, col_main = st.columns([1, 3])
with col_about:
    st.markdown("""
    <div class="about-box">
        <h3>ℹ️ About this App</h3>
        <p>This application provides <b>English → Hindi transliteration</b> using three approaches:</p>
        <ul>
            <li><b>🔡 LSTM Model</b> – Outputs via <i>Greedy</i> and <i>Beam Search</i>.</li>
            <li><b>🔠 Transformer Model</b> – Outputs via <i>Greedy</i> and <i>Beam Search</i>.</li>
            <li><b>🤖 Gemini API</b> – Uses Google’s Gemini AI to generate a single transliteration.</li>
        </ul>
        <p>👉 Compare sequence models, Transformer architectures, and modern LLM APIs side by side.</p>
    </div>
    """, unsafe_allow_html=True)

# Tabs
with col_main:
    tabs = st.tabs(["🔡 LSTM Model", "🔠 Transformer Model", "🤖 Gemini API"])

#tabs = st.tabs(["🔡 LSTM Model", "🔠 Transformer Model", "🤖 Gemini API"])

with tabs[0]:
    word = st.text_input("Enter word:", key="lstm_input")
    if st.button("🚀 Run LSTM", key="btn_lstm"):
        model, decoder_obj, src_vocab, tgt_vocab, src_idx2char, tgt_idx2char, max_len = load_lstm_model()
        encoded = encode_sequence_lstm(word, src_vocab, max_len)
        src_tensor = torch.tensor(encoded).unsqueeze(0)
        greedy_pred = decoder_obj.greedy_decode(src_tensor)
        beam_pred = decoder_obj.beam_search_decode(src_tensor, beam_width=5)
        st.markdown(f"<div class='result-box greedy'>Greedy: {decode_sequence_lstm(greedy_pred[1:], tgt_idx2char)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box beam'>Beam: {decode_sequence_lstm(beam_pred[1:], tgt_idx2char)}</div>", unsafe_allow_html=True)
        

with tabs[1]:
    word = st.text_input("Enter word:", key="trans_input")
    if st.button("🚀 Run Transformer", key="btn_trans"):
        model, src_vocab, tgt_vocab, src_idx2char, tgt_idx2char, max_src, max_tgt, decoder = load_transformer_model()
        greedy = transliterate_greedy(model, word, src_vocab, tgt_vocab, src_idx2char, tgt_idx2char, max_tgt, "cpu")
        beam = decoder.beam_search_decode(word, beam_width=5, max_length=max_tgt)
        st.markdown(f"<div class='result-box greedy'>Greedy: {greedy}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box beam'>Beam: {beam}</div>", unsafe_allow_html=True)
        
    
with tabs[2]:
    word = st.text_input("Enter word:", key="gem_input")
    if st.button("🚀 Run Gemini", key="btn_gem"):
        hindi = gemini_transliterate(word)
        st.markdown(f"<div class='result-box gemini'>Gemini Output: {hindi}</div>", unsafe_allow_html=True)
    

# Footer
st.markdown("<div class='footer'>✨ Built with ❤️ using LSTM + Transformer + Gemini ✨</div>", unsafe_allow_html=True)


