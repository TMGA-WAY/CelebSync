from dotenv import load_dotenv
import os

load_dotenv()
from langchain.llms import OpenAI

import streamlit as st

st.title("Langchain App")
input_text = st.text_input("Search")

open_ai_api_key = os.getenv("OPEN_AI_API_KEY")

### OPENAI LLM
## model_name="gpt-3.5-turbo",
llm = OpenAI(temperature=0.8, openai_api_key=open_ai_api_key)

if input_text:
    st.write(llm(input_text))
