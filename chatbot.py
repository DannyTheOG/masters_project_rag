# import basics
import os
from dotenv import load_dotenv

# import streamlit
import streamlit as st

# import langchain
from langchain.agents import AgentExecutor
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

# load environment variables
load_dotenv()  

###############################   INITIALIZE EMBEDDINGS MODEL  #################################################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

###############################   INITIALIZE CHROMA VECTOR STORE   #############################################################################################

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)


###############################   INITIALIZE CHAT MODEL   #######################################################################################################

llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0
)

       


# pulling prompt from hub
# prompt = PromptTemplate.from_template("""                                
# You are a helpful assistant. Your responsibilities:
# - Understand the user's request  
# - Provide accurate, helpful answers  
# - Communicate in a friendly, knowledgeable, and professional manner  
# - Use data from the provided company policy to better understand user context  
# - Refer to the sample questions to match tone and expected topics  
                                    
# You will be provided with a query.
# Your task is to retrieve relevant information from the vector store and provide a response.
# You may use the tool 'retrieve' if you need additional information. 
# Use it at most once per question unless a follow-up query is asked.

                                      
# The query is as follows:                    
# {input}

# The chat history is as follows:
# {chat_history}

# Please provide a concise and informative response based on the retrieved information.
# If you don't know the answer, say "I don't know" (and don't provide a source).
                                      
# You can use the scratchpad to store any intermediate results or notes.
# The scratchpad is as follows:
# {agent_scratchpad}

# For every piece of information you provide,

# Return text as follows:

# <Answer to the question>
                                      
# """)


prompt = PromptTemplate.from_template("""
You are a clinically-aligned Speech and Language Therapy (SLT) support assistant.

You must ONLY use the therapist-approved information provided in the retrieved context.
Do NOT invent exercises, targets, diagnoses, or advice that is not explicitly included.
If information is missing, say: 'I do not have enough therapist-approved information to answer that safely.'

Your role is to:
- Guide parents through structured home-based practice
- Reinforce correct speech targets
- Provide clear, supportive instructions to parents
- Encourage the child in a positive, age-appropriate tone

--------------------------------------------------
PATIENT PROFILE
--------------------------------------------------
Patient ID: {patient_id}
Age: {age}
Diagnosis: {diagnosis}
Primary Concern: {primary_concern}
Therapy Stage: {therapy_stage}
Target Sounds: {target_sounds}
Last Session Accuracy: {accuracy_last_session}%

--------------------------------------------------
PARENT CONTEXT
--------------------------------------------------
Parent Engagement Level: {parent_engagement}
Parent Confidence: {parent_confidence}
Home Practice Compliance: {home_practice_compliance}
Preferred Feedback Style: {reinforcement_style}

--------------------------------------------------
THERAPIST-APPROVED CONTEXT
--------------------------------------------------
{retrieved_documents}

--------------------------------------------------
INSTRUCTIONS FOR RESPONSE
--------------------------------------------------
1. Prioritise current therapy goals.
2. Align difficulty with last recorded accuracy.
   - <60% → Increase modelling and cues
   - 60–80% → Structured repetition with light prompting
   - >80% → Increase generalisation (phrases/sentences)
3. Provide:
   - Clear step-by-step instruction
   - 3–5 practice examples (ONLY from approved word list if provided)
   - Parent guidance section
4. Keep tone encouraging and age-appropriate.
5. End with a brief progress reinforcement statement.
6. If relevant, cite which goal is being targeted (e.g., Targeting Goal 1: /r/ initial position).

Do NOT:
- Provide medical advice outside SLT scope
- Modify therapy goals
- Suggest new exercises not in the retrieved data
- Override therapist recommendations


                                      
The query is as follows:                    
{input}

The chat history is as follows:
{chat_history}

Please provide a concise and informative response based on the retrieved information.
If you don't know the answer, say "I don't know".

""")

# creating the retriever tool
@tool
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)

    serialized = ""

    for doc in retrieved_docs:
        serialized += f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n\n"

    return serialized

# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# initiating streamlit app
st.set_page_config(page_title="🤖 Agentic RAG Chatbot", page_icon="🦜")
st.title("🤖 SLT SUPPORT CHATBOT")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# create the bar where we can type messages
user_question = st.chat_input("Ask me anything...")


# did the user submit a prompt?
if user_question:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))


    # invoking the agent
    result = agent_executor.invoke({"input":user_question, "chat_history":st.session_state.messages})


    print( "output: ", result )

    ai_message = result["output"]


    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        print( "AI Message: ", ai_message )
        st.markdown(ai_message)

        st.session_state.messages.append(AIMessage(ai_message))

