#! /usr/bin/env python

import os
from crewai_tools import PDFSearchTool, WebsiteSearchTool
#from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import tool
from crewai import Crew, Task, Agent
from langchain_groq import ChatGroq
import streamlit as st
import torch
import traceback
from dotenv import load_dotenv

###load environment variables
load_dotenv()

# API Key Configuration 
groq_api = os.getenv('GROQ_API_KEY')
#travily_api = os.getenv('TAVILY_API_KEY')

torch.classes.__path__ = [os.path.join(torch.__path__[0], 'torch', '_classes.py')]

# LLM Configuration

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
)

llm_image = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
)

# Tools Configuration
def setup_tools():
    # RAG Tool
    pdf_tool = PDFSearchTool(
        pdf=r"C:\Tasks\Multimodal_Agentic_RAG\Finance_CrewAI\Inputs\Fundamentals_Finance.pdf",
        config=dict(
            llm=dict(
                provider="groq",
                config=dict(
                    model="llama3-70b-8192",
                ),
            ),
            embedder=dict(
                provider="huggingface",
                config=dict(
                    model="BAAI/bge-small-en-v1.5",
                ),
            ),
        )
    )

    web_tool = WebsiteSearchTool(
        website="https://en.wikipedia.org/wiki/Investment",
        config=dict(
            llm=dict(
                provider="groq",
                config=dict(
                    model="llama3-70b-8192",
                ),
            ),
            embedder=dict(
                provider="huggingface",
                config=dict(
                    model="BAAI/bge-small-en-v1.5",
                ),
            ),
        )
    )

    # Web Search Tool
    #web_search_tool = TavilySearchResults(max_results=3)
    web_search_tool = DuckDuckGoSearchRun()

    return pdf_tool, web_tool, web_search_tool

# Router Tool
@tool
def router_tool(question):
    """Router Function"""
    pdf_keywords = ['PDF/Doc Summary', 'PDF/Doc Insights', 'PDF/Doc Uploaded/Attached', 'Financial Systems', 'Financial Markets', 'Finance Overview']
    web_keywords = ['Wikipedia', 'Investment', 'Investment and Risk', 'Investment Strategies', 'Webpage', 'Wikipage', 'Financial Investments', 'Types of Financial Investments']
    img_keywords = ['Image Summary', 'Image Insights', 'Image Uploaded', 'Image Attached']

    if any(keyword.lower() in question.lower() for keyword in pdf_keywords):
        return 'pdfvectorstore'
    elif any(keyword.lower() in question.lower() for keyword in web_keywords):
        return 'wikipagevectorstore'
    elif any(keyword.lower() in question.lower() for keyword in img_keywords):
        return 'imageanalysis'
    else:
        return 'websearch'


# Agent Definitions
def create_agents():
    Router_Agent = Agent(
        role='Router',
        goal='Route user question to a pdfvectorstore or wikipagevectorstore or imageanalysis or websearch.',
        backstory=(
            "You are an expert at routing a user question to a pdfvectorstore or wikipagevectorstore or imageanalysis or websearch based on the keywords provided in the tool."
            "Be flexible in interpreting keywords related to these topics."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    PDFRAG_Agent = Agent(
        role="PDFRAG",
        goal="Use retrieved information to answer the question from PDF vectorstore.",
        backstory=(
            "You are an assistant for question-answering tasks from PDF vectorstore. "
            "Provide clear, concise answers using retrieved context only from PDF vectorstore and nothing else."
            "Move to next Agent if question is not about PDF vectorstore."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    WebpageRAG_Agent = Agent(
        role='WikipageRAG',
        goal="Use retrieved information to answer the question from Wikipage vectorstore.",
        backstory=(
            "You are an assistant for question-answering tasks from Wikipage vectorstore. "
            "Provide clear, concise answers using retrieved context only from Wikipage vectorstore and nothing else."
            "Move to next Agent if question is not about Wikipage vectorstore."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    ImageAnalysis_Agent = Agent(
        role="ImageProcessor",
        goal="Read image and give insights or summary.",
        backstory=(
            "An expert in computer vision and image processing."
            "Provide clear, concise summary or insights from the image and nothing else."
            "Move to next Agent if question is not about Image analysis."
            ),
        multimodal=True,
        verbose=True,
        allow_delegation=False,
        llm=llm_image,
    )

    Websearch_Agent = Agent(
        role="WebSearcher",
        goal="Provide a comprehensive and accurate response from across internet or web and not from vectorstore.",
        backstory=(
            "You synthesize information from various sources to create a clear, concise, and informative answer to the user's question."
            "You can even write content based on user's input."
            "You should not use any vectorstore."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    return [Router_Agent, PDFRAG_Agent, WebpageRAG_Agent, ImageAnalysis_Agent, Websearch_Agent]

# Task Definitions
def create_tasks(agents, tools):
    pdf_tool, web_tool, web_search_tool = tools
    Router_Agent, PDFRAG_Agent, WebpageRAG_Agent, ImageAnalysis_Agent, Websearch_Agent = agents

    router_task = Task(
        description=(
            "Analyze the keywords in the question {question}. "
            "Decide whether it requires a pdfvectorstore search or wikipagevectorstore search or imageanalysis or websearch."
            "Strictly Return only 'pdfvectorstore' or 'wikipagevectorstore' or 'imageanalysis' or 'websearch' and nothing else."
            ),
        expected_output="'pdfvectorstore' or 'wikipagevectorstore' or 'imageanalysis' or 'websearch'",
        agent=Router_Agent,
        tools=[router_tool],
    )

    pdfrag_task = Task(
        description=(
            "Retrieve information for the question {question} from PDF vectorstore only and nothing else"
            "using pdfvectorstore based on router task."
            "Move to next Agent if question is not about PDF vectorstore."
            ),
        expected_output="Provide retrieved information from PDF vectorstore",
        agent=PDFRAG_Agent,
        tools=[pdf_tool],
    )

    webpagerag_task = Task(
        description=(
            "Retrieve information for the question {question} from Wikipage vectorstore only and nothing else"
            "using wikipagevectorstore based on router task."
            "Move to next Agent if question is not about Wikipage vectorstore."
            ),
        expected_output="Provide retrieved information from Wikipage vectorstore",
        agent=WebpageRAG_Agent,
        tools=[web_tool],
    )

    imageanalysis_task = Task(
        description="""
        1. Read the image from C:\\Tasks\\Multimodal_Agentic_RAG\\Finance_CrewAI\\Inputs\\Financial_Inclusion_Graph.png
        2. Give the summary/insights about the image only and nothing else
        3. Move to next Agent if question is not about Image Analysis
        """,
        expected_output="A summary/insight of image",
        agent=ImageAnalysis_Agent,
    )

    websearch_task = Task(
        description=(
            "Retrieve information or write content for the question {question} from across internet or web and not from vectorstore"
            "based on router task."
            ),
        expected_output="Provide retrieved information from across internet or websearch",
        agent=Websearch_Agent,
        tools=[web_search_tool],
    )

    return [router_task, pdfrag_task, webpagerag_task, imageanalysis_task, websearch_task]

def selected_crew(agents, tasks, question):
    # Create Crew
    rag_crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
    )

    result = rag_crew.kickoff(inputs={"question": question})

    return result

# Main RAG Function
#def run_rag_pipeline(question, history):
def run_rag_pipeline(question):
    # Setup tools
    pdf_tool, web_tool, web_search_tool = setup_tools()
    tools = (pdf_tool, web_tool, web_search_tool)

    # Create agents
    agents = create_agents()
    Router_Agent, PDFRAG_Agent, WebpageRAG_Agent, ImageAnalysis_Agent, Websearch_Agent = agents

    # Create tasks
    tasks = create_tasks(agents, tools)
    router_task, pdfrag_task, webpagerag_task, imageanalysis_task, websearch_task = tasks

    # Run the pipeline
    try:
        route_result = selected_crew([Router_Agent], [router_task], question)

        if "pdfvectorstore" in route_result :
            print("11111111111111111111111111111111111111111111111")
            print("route_result = = = = = = ", route_result)
            result = selected_crew([PDFRAG_Agent], [pdfrag_task], question)
        elif "wikipagevectorstore" in route_result :
            print("22222222222222222222222222222222222222222222222")
            print("route_result = = = = = = ", route_result)
            result = selected_crew([WebpageRAG_Agent], [webpagerag_task], question)
        elif "imageanalysis" in route_result :
            print("333333333333333333333333333333333333333333333333")
            print("route_result = = = = = = ", route_result)
            result = selected_crew([ImageAnalysis_Agent], [imageanalysis_task], question)
        else:
            print("44444444444444444444444444444444444444444444444444")
            print("route_result = = = = = = ", route_result)
            result = selected_crew([Websearch_Agent], [websearch_task], question)

        return result
    except Exception as e:
        return str(traceback.format_exc())


st.title("Finance Agentic RAG Chatbot")
st.markdown(
    "This is a multi-agent system using GROQ model that can perform various tasks such as search attached pdf/web/image for info, search or write across internet, for Financial Data."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Financial questions from Uploaded PDF, Image, Wikipedia page or Across Internet"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = run_rag_pipeline(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})