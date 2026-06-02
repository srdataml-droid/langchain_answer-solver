# import os
# from ollama import Client

# # 1. Set your API key (Get this from your ollama.com dashboard)
# os.environ["OLLAMA_API_KEY"] = "your_actual_api_key_here"

# # 2. Point the client to the online Ollama URL
# client = Client(host="https://ollama.com")

# # 3. Call a cloud model (e.g., deepseek-v3 or gpt-oss)
# response = client.chat(
#     model="deepseek-v3:cloud",  # Ensure the model ends with ':cloud'
#     messages=[
#         {"role": "user", "content": "Hello! What is your name?"}
#     ]
# )

# print(response["message"]["content"])

import os 
from tools import search_tool , wiki_tool
from dotenv import load_dotenv
import langchain
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()


class Response(BaseModel):
    topic : str
    summary : str
    sources : list[str]
    tools_used : list[str]
# llm1 = ChatOllama(model="deepseek-v3:cloud" , )

api_key = os.getenv("OLLAMA_API_KEYS")

# 3. Initialize ChatOllama with authentication headers
llm1 = ChatOllama(
    model="gpt-oss:120b-cloud", 

    # base_url="https://ollama.com/api", # Notice the /api path for Cloud endpoints
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {api_key}"
        }
    }   
)
parser = PydanticOutputParser(pydantic_object =  Response)

prompt = ChatPromptTemplate.from_messages(
    [
    ("system", 
     """
        You are a helpful assistant, made by samuel irenikase .\n{format_instructions} 
     """
     ),
    # ("user", "Summarize the following article: {article_url}")
    ("placeholder", "{chat_history}"),

    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
    # ("user", "Summarize the following article: {article_url}")
    
    ]
).partial(format_instructions = parser.get_format_instructions())

# llm2 = ChatOpenAI(model="gpt-4o")
# llm3 = ChatAnthropic(model="claude-2")

# response1 = llm1.invoke("Hello! What is your name?")
# print("Ollama Response:", response1.content)
tools = [search_tool , wiki_tool]
agents = create_tool_calling_agent(
    llm=llm1,
    tools=tools,
    prompt=prompt
)
agent_executor = AgentExecutor(agent=agents, tools=tools, verbose = True)
query = input("what do u need to know: ")
raw_response = agent_executor.invoke({"query" : query})
print("Raw Response:", raw_response)
try:
    # structured_response = parser.parse(raw_response.get("output")[0]["text"])
    # print("Structured Response:", structured_response)
    output_text = raw_response.get("output", "")
    structured_response = parser.parse(output_text.strip())
    
    print("\n--- Structured Response ---")
    print("Topic:", structured_response.topic)
    print("Summary:", structured_response.summary)
    print("Sources:", structured_response.sources)
    print("Tools Used:", structured_response.tools_used)
except Exception as e:
    print("Error parsing structured response:", e, "Raw Response : ", raw_response)





