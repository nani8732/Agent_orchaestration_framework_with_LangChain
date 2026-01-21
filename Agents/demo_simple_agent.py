#Step-1: Imports
from langchain_community.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
search_tool=DuckDuckGoSearchRun()
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
#Step-2: pull the request prompt from LangChain Hub
prompt=hub.pull("hwchase17/react")#pulls the standard ReAct agent prompt
#print(hub.pull("hwchase17/react"))

#Step-3: Create the react agent manually with the pulled prompt
agent=create_react_agent(llm=llm,
 tools=[search_tool],
  prompt=prompt
)
#Step-4: Wrap it with AgentExecutor
agent_executor=AgentExecutor(agent=agent,
 tools=[search_tool],
 verbose=True
)
#Step-5: Invoke
response=agent_executor.invoke({"input" : "3 ways to reach mumbai from delhi"})
print(response)