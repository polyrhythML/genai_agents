from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import asyncio
import uvicorn
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain_experimental.chat_models import Llama2Chat
from os.path import expanduser
from langchain.llms import OpenAI
from langchain_community.llms import LlamaCpp
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

class Item(BaseModel):
    """Base model for request body, expects 'input' as a field."""
    input: str

app = FastAPI()

def load_model():
    """
    Load the Llama2Chat model from a specified path.
    Returns the loaded model.
    """
    model_path = expanduser("~/work/amitbhatti/models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf")
    llm = LlamaCpp(model_path=model_path, n_ctx=2048,streaming=True)
    model = Llama2Chat(llm=llm)
    return model

def create_langchain(model):
    """
    Create a ConversationChain object with the loaded model and a specified memory.
    Returns the created ConversationChain object.
    """
    template_messages = [
        SystemMessage(content="You are an Assistant named BHAI which is bascially a slang in India for a friend who is pretty chill. So you too stay chill while answering queries"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(template_messages)
    chain = ConversationChain(
        llm=model,
        memory=ConversationBufferWindowMemory(k=6)
    )
    return chain

# Load the model and create the ConversationChain object
model = load_model()
langchain = create_langchain(model)

@app.post("/predict")
async def predict(item: Item):
    """
    Endpoint to generate a response from the ConversationChain object.
    Expects a POST request with 'input' in the request body.
    Returns the generated response.
    """
    response = langchain.predict(input=item.input)
    return {"response": response}

if __name__ == "__main__":
    # Run the application using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
