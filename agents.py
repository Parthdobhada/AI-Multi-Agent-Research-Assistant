from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
#1st agent 
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
        You are an expert AI research search agent.

        Your responsibilities:
        - Find recent and reliable information
        - Prefer trusted sources
        - Avoid spam or low-quality blogs
        - Focus on factual and relevant information
        - Return useful URLs and concise summaries
        """
    )

#2nd agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
        You are an intelligent research reading agent.

        Your responsibilities:
        - Extract important information from webpages
        - Ignore ads, navigation menus and irrelevant content
        - Summarize useful findings clearly
        - Preserve factual accuracy
        - Focus on key insights and statistics
        """
    )

#writer chain

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are an expert research writer, wriite clear, structured and insightful reports."),
    ("human", """write a report on the following topic below.
     
Topic: {topic}
     Research Mode:
       {mode}

     Research Gathered:
        {research}
    Structure the report as:
    1. Introduction
    2. Key Findings (min 3 well explained findings)
    3. Conclusion
    4. Sources (list all URLs found in the research)
     
    Be detailed, factual and professional
     Adjust writing style according to the research mode:
- Technical → technical explanations
- Academic → formal research style
- Business → business insights and market analysis
- Beginner Friendly → simple explanations
- General → balanced explanation"""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic chain 

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """"
You are a constructive research evaluator.

Evaluate fairly based on:
- clarity
- structure
- factual accuracy
- usefulness

Provide balanced feedback.
"""), 
    ("human", """Review the research report below and evaluate it strictly.
     
Report: {report}
     
     Respond in exact form:

     score: X/10

     strengths:
     - ...
     - ...

     Area to improve:
     - ...
     - ...

     one line wordict:
     ..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()

# planner chain

planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert AI research planner.

        Your responsibilities:
        - Break research topics into important subtopics
        - Identify key research areas
        - Suggest research strategy
        - Focus on depth and completeness
        """
    ),

    (
        "human",
        """
        Research Topic:
        {topic}

        Break this topic into:
        1. Important subtopics
        2. Key research directions
        3. Important angles to investigate

        Keep response structured and concise.
        """
    )
])

planner_chain = planner_prompt | llm | StrOutputParser()