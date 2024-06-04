import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, FalconForCausalLM, Conversation
import streamlit as st
from peft import PeftConfig, PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

@st.cache_resource
def load_model():
    base_model_name = "tiiuae/falcon-11B"
    peft_model_path = "/home/michael/robot/tuned_model2"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, max_model_length=8192)
    base_model = FalconForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    peft_config = PeftConfig.from_pretrained(peft_model_path)
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    
    return tokenizer, model

tokenizer, model = load_model()

def clean_response(response):
    response = re.sub(r'([^\w\s])\1+', r'\1', response)
    response = re.sub(r'(\b\w+\b)(?:\s+\1)+', r'\1', response)
    response = re.sub(r'(\b\d+\b)(?:\s+\1)+', r'\1', response)
    response = re.sub(r'(\b.+?\b)(?:\s+\1)+', r'\1', response)
    return response

st.title("TurnyBot")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Enter your prompt:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=min(tokenizer.model_max_length, 512))
    except OverflowError as e:
        st.error(f"Error tokenizing input: {e}")
        inputs = tokenizer("", return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.9
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    cleaned_response = clean_response(generated_text)
    cleaned_response = re.sub(r'[^\w\s,.\'\"!?;:()\-]', '', cleaned_response)
    
    st.session_state.messages.append({"role": "assistant", "content": cleaned_response})

    with st.chat_message("assistant"):
        st.write(cleaned_response)
