import os
import json
import streamlit as st
from rental_agent import agent_with_history, parse_weight_to_tons
from create_vector_db_and_qa_chain import get_vector_db, create_qa_chain_with_memory
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import random


# Create a random number to be used for the agent memory
if "seed" not in st.session_state:
    st.session_state.seed = 42

random.seed(st.session_state.seed)
random_session_number = random.randint(10**49, 10**50 - 1)

# configuring streamlit page settings
st.set_page_config(page_title="Chani.AI Agent - Forklift Rental Assistant", layout="centered")

st.title("🤖 Chani.AI Agent - Forklift Rental Assistant")
st.markdown("Your AI-powered agent to help you specify and book the right forklift.\n\nChani.AI will guide you through 7 key questions and complete the booking when ready.")

# initialize chat session in streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   
     
# display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# input field for user's message
user_prompt = st.chat_input("Type your forklift rental request...")        

if user_prompt:
    # add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # send user's message to GPT-4o and get a response
    response = agent_with_history.invoke(
    {"input": user_prompt,
     "agent_scratchpad": []},
    # "I want a forklift for 3 months, 2t load, 1200x1200x1000, sealed surface, turning radius 2150mm, indoor and outdoor use"
    config={"session_id": f"ABC-{random_session_number}"}
    )

    assistant_response = response['output']
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # display GPT-4o's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    action_output = {}
    for i, (action, output) in enumerate(response.get("intermediate_steps", [])):
        # st.markdown(f"**Step {i+1}: Tool `{action.tool}`**")
        # st.code(f"Input: {action.tool_input}\nOutput: {output}")
        action_output[str(action.tool)] = output
    
    # st.markdown(action_output)    

    if ('print_finish' in action_output.keys()) and action_output['print_finish']:
        st.markdown(action_output['print_finish']) 
        st.session_state.chat_history.append({"role": "assistant", "content": action_output['print_finish']})
           
    
    if ('get_max_lifting_weight' in action_output.keys()) and action_output['get_max_lifting_weight']:
        max_load_float = parse_weight_to_tons(action_output['get_max_lifting_weight'])
        weight_message = f'🏋️**The maximum lifting weight is {max_load_float} ton.**'
        st.markdown(weight_message) 
        st.session_state.chat_history.append({"role": "assistant", "content": weight_message})
        
        # Get the previously created vector store (pinecone database)
        vector_db = get_vector_db()

        prompt_template = '''
            You are a helpful AI assistant to answer queries about renting forklifts from our company.
            Answer my questions based on the ifnormation provided and the older conversation. Do not make up answers.
            If you do not know the answer to a question, just say "I don't know".

            You have two tasks to do given the information and requirements obtained from the user:
                    
            1) find the forklift that meet the user's requirements.
           
            2) if there is any forklift, a price quote for that recommended Forklift.

            NOTE that the forklift maximum load capacity MUST be EQUAL or if NOT, higher than the maximum lifting weight specified by user, 
            DO NOT use italic or stick words together in your answer.

            SAMPLE QUOTE:            
            For a rental duration of 1 month (28+ days), the weekly rate is 126.00.
            Therefore, the total cost for 1 month would be 126.00 x 4 = $504.00. P
            lease note that this price does not include delivery charges to Adelaide CBD.

            {context}

            Given the following conversation and a follow up question, answer the question.

            {chat_history}

            question: {question}
            '''

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "chat_history", "question"]
        )

        LLM = ChatOpenAI(model_name="gpt-4o", temperature=0) # no creativity we want to be accurate (temp=0)

        MEMORY = ConversationBufferMemory(
                                memory_key="chat_history",
                                max_len=50, return_messages=True,
                                output_key='answer')

        # Create question-answering chain with memory
        qa_chain_with_memory = create_qa_chain_with_memory(vectorstore=vector_db, 
                                                           memory=MEMORY, 
                                                           llm=LLM, prompt=PROMPT)

        # First query
        query = f""" A recommended forklif with a quote for that given the following info and specifications:
                        {action_output['print_finish']}. The maximum lifting weight is {max_load_float} ton"""
        answer = qa_chain_with_memory.invoke({'question':query, 
                                              'chat_history': MEMORY.chat_memory.messages})
        
        recommended_forklif = answer['answer']
        st.markdown(recommended_forklif) 
        st.session_state.chat_history.append({"role": "assistant", "content": recommended_forklif})

        st.session_state.seed = st.session_state.seed + 10
        action_output = {}

    
