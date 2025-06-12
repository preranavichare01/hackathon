import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import chromadb
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# ------------------ Load ChromaDB Collection ------------------
client = chromadb.Client()
collection = client.get_or_create_collection("energial_data")

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    data = {}
    for file in ["energy_consumption.csv", "energy_generation.csv", "weather.csv", "building_information.csv"]:
        data[file] = pd.read_csv(f"data/{file}")
    return data

data = load_data()

# ------------------ Setup NVIDIA Mistral Agent ------------------
agent = ChatNVIDIA(
    model="mistralai/mistral-7b-instruct-v0.3",
    api_key="",  # Only needed if running outside NGC
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)

def ask_agent(query):
    for chunk in agent.stream([{"role": "user", "content": query}]):
        yield chunk.content

# ------------------ Dashboard Layout ------------------
st.title("EnergiAI 🌱 – Carbon-Aware Energy Dashboard")
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧠 AI Assistant", "📁 Datasets"])

# ------------------ Charts ------------------
with tab1:
    st.header("Energy Usage per Building")
    fig1, ax1 = plt.subplots()
    df1 = data["energy_consumption.csv"]
    df1.groupby("building_id")["energy_usage"].sum().plot(kind="bar", ax=ax1)
    st.pyplot(fig1)

    st.header("Estimated Carbon Emissions")
    carbon_factors = {"coal": 820, "natural_gas": 490, "solar": 45, "wind": 12}
    df2 = data["energy_consumption.csv"]
    df2["carbon_emission"] = df2.apply(lambda r: carbon_factors.get(r["energy_source"], 100) * r["energy_usage"], axis=1)
    fig2, ax2 = plt.subplots()
    df2.groupby("energy_source")["carbon_emission"].sum().plot(kind="bar", ax=ax2, color='orange')
    st.pyplot(fig2)

# ------------------ Agent Chat ------------------
with tab2:
    st.header("Agentic AI Assistant 🤖")
    query = st.text_input("Ask your AI assistant (e.g., How to reduce carbon usage?)")
    if query:
        st.markdown("**Answer:**")
        response = ""
        for chunk in ask_agent(query):
            response += chunk
            st.write(response)

# ------------------ View Datasets ------------------
with tab3:
    for name, df in data.items():
        st.subheader(name)
        st.dataframe(df)
