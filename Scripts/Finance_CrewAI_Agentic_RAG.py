#! /usr/bin/env python

import os
import gradio as gr
from crewai_tools import PDFSearchTool, WebsiteSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
#from crewai_tools import tool
from crewai import Crew, Task, Agent, Process
from langchain_groq import ChatGroq
from dotenv import load_dotenv

###load environment variables
load_dotenv()

# API Key Configuration 
groq_api = os.getenv('GROQ_API_KEY')
travily_api = os.getenv('TAVILY_API_KEY')

# LLM Configuration

llm = ChatGroq(
    name="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000
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
                    model="llama3-8b-8192",
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
                    model="llama3-8b-8192",
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
    web_search_tool = TavilySearchResults(max_results=3)

    return pdf_tool, web_tool, web_search_tool

'''# Router Tool
@tool
def router_tool(question):
    """Router Function"""
    pdf_keywords = ['PDF/Doc Summary', 'PDF/Doc Insights', 'PDF/Doc Uploaded/Attached', 'Financial Systems', 'Financial Markets', 'Finance Overview']
    web_keywords = ['Wikipedia', 'Investment', 'Investment and Risk', 'Investment Strategies', 'Webpage']
    img_keywords = ['Image Summary', 'Image Insights', 'Image Uploaded', 'Image Attached']

    if any(keyword.lower() in question.lower() for keyword in pdf_keywords):
        return 'pdf_vectorstore'
    elif any(keyword.lower() in question.lower() for keyword in web_keywords):
        return 'webpage_vectorstore'
    elif any(keyword.lower() in question.lower() for keyword in img_keywords):
        return 'image_analysis'
    else:
        return 'websearch'
'''

# Agent Definitions
def create_agents():
    Router_Agent = Agent(
        role='Router',
        goal='Route user question to a pdf_vectorstore or webpage_vectorstore or image_analysis or websearch',
        backstory=(
            "You are an expert at routing a user question to a pdf_vectorstore or webpage_vectorstore or imaage_analysis or websearch. "
            "Use the pdf_vectorstore for questions related to PDF Retrieval-Augmented Generation based on keywords ['PDF/Doc Summary', 'PDF/Doc Insights', 'PDF/Doc Uploaded/Attached', 'Financial Systems', 'Financial Markets', 'Finance Overview']. "
            "Use the webpage_vectorstore for questions related to Webpage Retrieval-Augmented Generation based on keywords ['Wikipedia', 'Investment', 'Investment and Risk', 'Investment Strategies', 'Webpage'] "
            "Use the image_analysis for questions related to image summary/insights based on keywords ['Image Summary', 'Image Insights', 'Image Uploaded', 'Image Attached']"
            "Use the normal internet search if no keywords available from the above list in the question."
            "Be flexible in interpreting keywords related to these topics."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    PDFRAG_Agent = Agent(
        role="PDFRAG",
        goal="Use retrieved information to answer the question from PDF vectorstore",
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
        role='WebpageRAG',
        goal="Use retrieved information to answer the question from Webpage vectorstore",
        backstory=(
            "You are an assistant for question-answering tasks from Webpage vectorstore. "
            "Provide clear, concise answers using retrieved context only from Webpage vectorstore and nothing else."
            "Move to next Agent if question is not about Webpage vectorstore."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    ImageAnalysis_Agent = Agent(
        role="ImageProcessor",
        goal="Read image and give insights or summary",
        backstory=(
            "An expert in computer vision and image processing."
            "Provide clear, concise summary or insights from the image and nothing else."
            "Move to next Agent if question is not about Image analysis."
            ),
        multimodal=True,
        llm=llm
    )

    Websearch_Agent = Agent(
        role="WebSearcher",
        goal="Provide a comprehensive and accurate response from across internet or web and not from vectorstore",
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

    '''router_task = Task(
        description=(
            "Analyze the keywords in the question {question}. "
            "Decide whether it requires a pdf_vectorstore search or webpage_vectorstore search or image_analysis or websearch."
        ),
        expected_output="Return 'pdf_vectorstore' or 'webpage_vectorstore' or 'image_analysis' or 'websearch'",
        agent=Router_Agent,
        tools=[router_tool],
    )'''

    pdfrag_task = Task(
        description=(
            "Retrieve information for the question {question} from PDF vectorstore only and nothing else"
            "using pdf_vectorstore based on router task."
            "Move to next Agent if question is not about PDF vectorstore."
        ),
        expected_output="Provide retrieved information from PDF vectorstore",
        agent=PDFRAG_Agent,
        tools=[pdf_tool],
    )

    webpagerag_task = Task(
        description=(
            "Retrieve information for the question {question} from Webpage vectorstore only and nothing else"
            "using webpage_vectorstore based on router task."
            "Move to next Agent if question is not about Webpage vectorstore."
        ),
        expected_output="Provide retrieved information from Webpage vectorstore",
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

    #return [router_task, pdfrag_task, webpagerag_task, imageanalysis_task, websearch_task]
    return [pdfrag_task, webpagerag_task, imageanalysis_task, websearch_task]


# Main RAG Function
def run_rag_pipeline(question, history):

    # Setup tools
    pdf_tool, web_tool, web_search_tool = setup_tools()
    tools = (pdf_tool, web_tool, web_search_tool)

    # Create agents
    agents = create_agents()

    # Create tasks
    tasks = create_tasks(agents, tools)

    # Create Crew
    rag_crew = Crew(
        agents=agents[1:],
        tasks=tasks,
        manager_agent=agents[0],
        manager_llm=llm,
        process=Process.hierarchical,
        verbose=True,
    )

    # Run the pipeline
    try:
        result = rag_crew.kickoff(inputs={"question": question})
        history.append(["user", question])
        history.append(["assistant", result])
        return "", history
    except Exception as e:
        return f"An error occurred: {str(e)}", history

def gradio_interface(query, history):
    if history is None:
        history = []
    if not query:
        return "Please enter a question.", history
    result, history = run_rag_pipeline(query, history)
    return result, history

def create_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown(
        """
        # Financial Agentic RAG ChatBot
        Ask questions about Finance, Summary, Insights, Investment, Graphs, Writing Content, etc., The system uses a multi-agent approach to retrieve and provide information.
        """
        )
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Ask your question about Finance, Summary, Insights, Investment, Graphs, Writing Content, etc.")
        clear = gr.ClearButton([msg, chatbot])

        msg.submit(gradio_interface, [msg, chatbot], [msg, chatbot])

    return demo

# Launch the Gradio App
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch()