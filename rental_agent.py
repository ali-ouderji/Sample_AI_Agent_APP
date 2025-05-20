from langchain.chat_models import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools import tool
import os
import json
import streamlit as st
import re
# from dotenv import load_dotenv

# Load environment variables
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def parse_weight_to_tons(text: str) -> float:
    """
    Extracts weight from input text and converts it to tons.
    Handles formats like '4.5 t', '4500 kg', '4.5 tons', etc.
    """
    text = text.lower()
    match = re.search(r'(\d+(\.\d+)?)\s*(t|tons?|kg)', text)

    if not match:
        raise ValueError("No valid weight found in input.")

    value = float(match.group(1))
    unit = match.group(3)

    if 'kg' in unit:
        value = value / 1000.0  # convert kg to tons

    return value

@tool
def print_finish(data: str) -> str:
    """Finalize the forklift rental request. Data should be a structured summary string."""
    return f"✅ Rental info captured and ready: {data}"

@tool
def get_max_lifting_weight(text: str) -> float | str:
    """
    After using 'print_finish' tool and finishing the forklift rental request,
    extract ONLY the maximum lifting weight value. 
    Respond with just the weight value.
    Example expected output: '2t' or '2000kg'
    """

    # Look for patterns like "2t", "2000kg", "1.5 tons", etc.
    patterns = [
        r"(\d+(?:\.\d+)?\s?t)",          # e.g., 2t or 2.5 t
        r"(\d+(?:\.\d+)?\s?tons?)",      # e.g., 2 tons
        r"(\d+(?:\.\d+)?\s?kg)",         # e.g., 2000kg
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            max_load = match.group(1)
            return max_load

    return "No maximum lifting weight found in the input."

# Tools
tools = [print_finish, get_max_lifting_weight]

# LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o")

# REQUIRED: Prompt that includes `agent_scratchpad`
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
You are a helpful assistant called Chani.AI.

After reciving user query do the following tasks one by one: 

- First task: check if user query contain the following answers:

1) Maximum lifting weight?
“What is the heaviest load you wll need to lift?” Sample User Answer: 2t 

2) Maximum load size?
“Roughly what are the maximum load dimensions? (Length x Width x Height)” Sample User Answer: 1200x1200x1000

3) Indoor, outdoor, or both?
“Will you be using the forklift indoors, outdoors, or a mix of both?” Sample User Answer: Both 

4) Ground conditions?
“Is the ground flat and sealed, or will you be on gravel, dirt, or uneven surfaces?” Sample User Answer: sealed surface 

5) Turning space or access limits?
“Are there any tight turning areas, low clearances, or confined spaces we should know about?”  Sample User Answer: my turning radius cant be more than 2150mm quite a tight spot 

6) Rental duration?
“How long do you need the forklift for? (e.g., 1 week, 1 month, ongoing)” Sample User Answer: 3 months 

7) Delivery location?
“Where are you located or where do you need the machine delivered to?” Sample User Answer: Perth Bibra Lake just around the corner

- Second task: summerize the answers if all are availble. Otherwise, ask the above questions from the user in one message 
  and then make sure you have all information (answers to all questions). 
  If the user answer, does not include all answers, ask for the missing answer.
  Do this in a loop until you have all information

IMPORTANT: Once all required answers are collected, call the `print_finish` tool with a summary."""
    ),
    MessagesPlaceholder(variable_name="history"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

# Create the agent
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

# Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, 
                               verbose=True, max_iterations=3,
                               return_intermediate_steps=True)

# Memory
chat_map = {}
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_map:
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]

# Wrap with memory support
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history=get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)
