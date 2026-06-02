import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser  # 1. Import the text parser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_ollama import ChatOllama

load_dotenv()
api_key = os.getenv("OLLAMA_API_KEY")

llm1 = ChatOllama(
    model="gpt-oss:120b-cloud", 
    base_url="https://ollama.com",
    client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}}
)

# 2. Initialize the Text Output Parser
parser = StrOutputParser()

# 3. Define prompt BEFORE building the agent (Notice: {format_instructions} removed)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates clean Python code or text responses."),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 4. Initialize Agent and Executor
agents = create_tool_calling_agent(
    llm=llm1,
    tools=[],
    prompt=prompt
)
agent_executor = AgentExecutor(agent=agents, tools=[], verbose=True)

# 5. Run the query
user_query = input("Enter your query: ")
raw_response = agent_executor.invoke({"query": user_query})

print("Raw Response:", raw_response)

# 6. Extract the output text and write it straight to a python file
try:
    # Extract the raw string output text from the executor dictionary
    raw_output_text = raw_response.get("output", "")
    
    # Process it through the StrOutputParser to get clean text
    clean_text = parser.parse(raw_output_text)
    print("\n--- Clean Parsed Text ---\n", clean_text)
    
    # Save the output directly into a new Python file
    output_filename = "generated_output.py"
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(clean_text)
        
    print(f"\nSuccessfully parsed and saved code to '{output_filename}'!")

except Exception as e:
    print("Error parsing structured response:", e)
