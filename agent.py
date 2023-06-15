from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.tools import HumanInputRun

from tools import retrieval_tool, sentiment_tool, ner_tool, time_series_tool

def get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)

openai_api_key='OpenAI API key'
date_des = '''
The input of month should be number of two digit. For example, February -> 02.
If the year is not specified, then input "all" as the value of year.
If the month is not specified, then input "all" as the value of month.
'''
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
tools = [
    StructuredTool.from_function(
        func=retrieval_tool,
        name = "retrieval",
        description=f"Useful for when you need to retrieve information on specific date. {date_des}"
    ),
    StructuredTool.from_function(
        func=sentiment_tool,
        name = "sentiment analysis",
        description=f"Useful for when you need to know the sentiment on specific date. {date_des}"
    ),
    StructuredTool.from_function(
        func=ner_tool,
        name = "named entity recognition",
        description=f"Useful for when you need to know the most popular people on specific date. {date_des}"
    ),
    StructuredTool.from_function(
        func=time_series_tool,
        name = "social voice trend",
        description="Useful for when you need to know the trend of social volume in date interval. The input of date should be number of six digit. For example, February, 2023 -> 202302."
    ),
    HumanInputRun(input_func=get_input)
]
agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

prompt = '''
I want to know the most popular people in Feb, 2023. 
Also, please tell me some information about the person at that time.
Then, tell me the sentiment toward him.
Last, plot his trend of social volume from May, 2022 to May, 2023.
Reply in traditional Chinese.
'''
agent.run(prompt)

