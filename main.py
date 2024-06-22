from dotenv import load_dotenv
import os

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

load_dotenv()
open_ai_api_key = os.getenv("OPEN_AI_API_KEY")
st.title("Celebrity Search")
input_text = st.text_input("Search")

# Prompt template
input_prompt = PromptTemplate(
    input_variables=['name'],
    template='tell me about {name}'
)

##memory
person_memory = ConversationBufferMemory(input_key="name", memory_key="chat_history")
dob_memory = ConversationBufferMemory(input_key="person", memory_key="chat_history")
descr_memory = ConversationBufferMemory(input_key="dob", memory_key="chat_history")

# OPENAI LLM
llm = OpenAI(temperature=0.8, openai_api_key=open_ai_api_key)
chain = LLMChain(llm=llm, prompt=input_prompt, verbose=True, output_key='person', memory=person_memory)

## second chain
second_prompt = PromptTemplate(
    input_variables=['person'],
    template="when was {person} born?"
)
chain2 = LLMChain(llm=llm, prompt=second_prompt, verbose=True, output_key='dob', memory=dob_memory)

## third chain
third_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {dob} in the world."
)
chain3 = LLMChain(llm=llm, prompt=third_prompt, verbose=True, output_key='description', memory=descr_memory)

parent_chain = SequentialChain(
    chains=[chain, chain2, chain3], input_variables=['name'], output_variables=['person', 'dob', 'description'],
    verbose=True)

if input_text:
    parent_chain({'name': input_text})
    # st.write(parent_chain({'name': input_text}))
    with st.expander('Person Name'):
        st.info(person_memory.buffer)
    with st.expander('Born'):
        st.info(dob_memory.buffer)
    with st.expander('Major Events'):
        st.info(descr_memory.buffer)
