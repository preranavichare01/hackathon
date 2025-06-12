from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import LLMMathChain
import pandas as pd

# Load datasets
energy = pd.read_csv("data/energy_consumption.csv")
generation = pd.read_csv("data/energy_generation.csv")
weather = pd.read_csv("data/weather.csv")
buildings = pd.read_csv("data/building_information.csv")

# Define Mistral agent
llm = ChatNVIDIA(
    model="mistralai/mistral-7b-instruct-v0.3",
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    api_key="your_nvidia_api_key"
)

# Define dataset tools
def get_energy_summary(_):
    return energy.describe().to_string()

def get_generation_summary(_):
    return generation.describe().to_string()

tools = [
    Tool(name="Energy Summary", func=get_energy_summary, description="Summary of energy consumption data"),
    Tool(name="Generation Summary", func=get_generation_summary, description="Summary of energy generation data")
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ask Mistral to generate chart code
query = """Using the 4 datasets available, generate Python code to create Streamlit charts 
that show energy usage over time, most used energy source, and carbon emission trends. 
Return only the code block."""
response = agent.run(query)

print("🧠 Mistral agent responded with Streamlit code:\n")
print(response)
