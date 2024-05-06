from fastapi import FastAPI, Body
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.agents import Tool
from langchain.tools import tool
from langchain.tools.render import format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
import requests
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from pydantic import BaseModel, Field
import datetime
import wikipedia
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import Tool
from langchain.tools import tool
from langchain.tools.render import format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser

import uvicorn

# Define the request body model
class QAInput(BaseModel):
    input: str

# Define the response model
class QAResponse(BaseModel):
    output: str

# Initialize FastAPI app
app = FastAPI()

import wikipedia

@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


# Use Duck Duck Go for searching web for real-time queries, e.g. current affairs.
# Adding a search tool

search = DuckDuckGoSearchAPIWrapper()
search_tool = Tool(name="Current Search",
                   func=search.run,
                   description="Useful when you need to answer questions about nouns, current events or the current state of the world."
                   )

tools = [search_tool, search_wikipedia]

#model = LlamaCpp(
#    model_path="../models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf",
#    n_gpu_layers=1,
#    n_batch=512,
#    f16_kv=True,
#    verbose=True,
#    streaming=True)

model = ChatOpenAI(model="gpt-3.5-turbo", 
                temperature=0, 
                streaming=True, 
                api_key="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")


def create_qa_agent():
    # Your existing code here...
    # ...
    # functions = [format_tool_to_openai_function(f) for f in tools]

    # Binding tools functions with OpenAI model
    #model = ChatOpenAI(model="gpt-3.5-turbo", 
    #                temperature=0, 
    #                streaming=True, 
    #                api_key="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")

    # Adding ConversationBuffer
    memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")

    # Defining Chatprompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful but sassy assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        # MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Agent Scratchpad is the intermediate thinking the Agent does
    # Runnable Passthrough provides intermediate inputs to the conversation
    chain = RunnablePassthrough.assign(
        agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
    ) | prompt | model | OpenAIFunctionsAgentOutputParser()
     
    # Simple Chain Experiment
    # chain = prompt | model | StrOutputParser()

    # AgentExecutor takes in lang chain
    qa = AgentExecutor(agent=chain, tools=tools, verbose=True, memory=memory)
    
    return qa

# Store the qa agent in a global variable so it's not recreated every time the endpoint is called
qa_agent = create_qa_agent()

@app.put("/ask", response_model=QAResponse)
async def ask_question(input: QAInput = Body(...)):
    result = qa_agent.invoke({"input": input.input})
    print(result)
    answer = result['output']
    return QAResponse(output=answer)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
