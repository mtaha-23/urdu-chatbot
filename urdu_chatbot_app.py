import streamlit as st
import torch
import pickle
from urdu_chatbot import UrduChatbot, UrduTokenizer


st.set_page_config(page_title="Urdu Chatbot", page_icon="💬", layout="centered")

st.title("🤖 Urdu Conversational Chatbot")
st.caption("Transformer Encoder–Decoder • PyTorch")

# RTL/Urdu rendering styles
st.markdown(
    """
<style>
.urdu-text { direction: rtl; text-align: right; font-family: 'Noto Sans Arabic','Arial',sans-serif; font-size: 16px; line-height: 1.6; }
.stTextInput > div > div > input { direction: rtl; text-align: right; }
</style>
""",
    unsafe_allow_html=True,
)


class _TokenizerFixUnpickler(pickle.Unpickler):
    # Remap pickled references to __main__.UrduTokenizer -> urdu_chatbot.UrduTokenizer
    def find_class(self, module, name):
        if (module == '__main__' or module.endswith('.ipynb')) and name == 'UrduTokenizer':
            return UrduTokenizer
        return super().find_class(module, name)


@st.cache_resource(show_spinner=False)
def load_chatbot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('tokenizer.pkl', 'rb') as f:
        try:
            tokenizer = _TokenizerFixUnpickler(f).load()
        except Exception:
            f.seek(0)
            tokenizer = pickle.load(f)
    bot = UrduChatbot('best_model.pth', tokenizer, device)
    return bot


with st.spinner("Loading model…"):
    chatbot = load_chatbot()

decoding_method = st.radio("Decoding method", ["greedy", "beam"], horizontal=True)
beam_size = 3
if decoding_method == "beam":
    beam_size = st.slider("Beam size", 2, 10, 3)

user_input = st.text_input("اپنا پیغام لکھیں:", placeholder="آپ کیسے ہیں؟")

if st.button("Send", type="primary") and user_input.strip():
    if decoding_method == "greedy":
        response = chatbot.chat(user_input.strip(), method="greedy")
    else:
        response = chatbot.generate_response(user_input.strip(), method="beam", beam_size=beam_size)

    st.markdown(f"**You:**\n\n<span class='urdu-text'>{user_input.strip()}</span>", unsafe_allow_html=True)
    st.markdown(f"**Bot:**\n\n<span class='urdu-text'>{response}</span>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Place this file alongside best_model.pth, tokenizer.pkl, and urdu_chatbot.py. Run: streamlit run urdu_chatbot_app.py")


