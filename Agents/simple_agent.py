import requests
import os
from langchain_community.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import numexpr
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

search_tool=DuckDuckGoSearchRun()
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool 
def get_weather_data(city:str)->dict:
    """
    This tool fetches the weather data of city.
    """
    data=requests.get(
        "http://api.weatherstack.com/current",
        params={'access_key':os.getenv("WEATHERSTACK_API_KEY"), 'query':city})
    if 'current' not in data:
        raise RuntimeError(data)
    else:
        return{
            'city':city,
            "temperature":data["current"]["temperature"],
            "condition":data["current"]["weather_descriptions"][0]
        }

@tool 
def calculator(expression:str)->float:
    """Evaluate a math expression safely."""
    return float(numexpr.evaluate(expression))
prompt=hub.pull("hwchase17/react")
agent=create_react_agent(llm=llm,
 tools=[search_tool,get_weather_data],
  prompt=prompt
)
agent_executor=AgentExecutor(agent=agent,
 tools=[search_tool,get_weather_data],
 verbose=True
)
response=agent_executor.invoke({"input" : "3 ways to reach mumbai from delhi"})
print(response)