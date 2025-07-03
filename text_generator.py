import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

@st.cache_resource
def load_model(model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    return model, tokenizer


def generate_text(prompt, max_length: int = 150):
    """Generate text continuing the given prompt using GPT‑2."""
    model, tokenizer = load_model()

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# --------------------------- UI --------------------------- #

st.title("🧠 GPT‑2 Text Generator")
st.subheader("Generate a paragraph on any topic and download it as a text file.")

user_input = st.text_input("Enter a topic or prompt", "The future of artificial intelligence")

if st.button("Generate"):
    if not user_input.strip():
        st.warning("Please enter a prompt…")
    else:
        with st.spinner("Generating…"):
            generated = generate_text(user_input)

        st.write("### 📝 Output:")
        st.write(generated)

        # ------- Download button ------- #
        st.download_button(
            label="💾 Download as .txt",
            data=generated,
            file_name="generated_text.txt",
            mime="text/plain",
        )
