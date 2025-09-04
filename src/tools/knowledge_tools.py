from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
)
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper,
)
from src.models.llm_provider import get_llm_knowledge_base
from src.utils.rate_limiter import rate_limiter
from langchain_core.tools import tool

# Rate limits according to API docs
ARXIV_DELAY = 3.0 
WIKIPEDIA_DELAY = 3.0 


arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1, 
    arxiv_search=None,
    arxiv_exceptions=None
)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, wiki_client=None)
wikipedia = WikipediaQueryRun(
    api_wrapper=wikipedia_wrapper,
    description="Search for information on a given topic using Wikipedia"
)


@rate_limiter.rate_limit("arxiv", ARXIV_DELAY)
def search_arxiv(query: str) -> str:
    """Search for information on a given topic using Arxiv"""
    return arxiv.run(query)

@rate_limiter.rate_limit("wikipedia", WIKIPEDIA_DELAY)
def search_wikipedia(query: str) -> str:
    """Search for information on a given topic using Wikipedia"""
    return wikipedia.invoke(query)


@tool
def llm_knowledge(caption: str, question: str) -> str:
    """Uses the LLM to generate background knowledge, definitions, or common-sense context about key concepts found in the image caption and question."""
    prompt_template = f"""
     Please generate the background knowledge based on the key words in the context and question.
        ======
        Context: A snowboarder making a run down a powdery slope on a sunny day.
        Question: What is this man on?
        LLM_Knowledge: A snowboarder is a person who rides a snowboard. Snowboarding is a winter sport that involves riding down a snow-covered slope on a snowboard. xxxxxx
        A powdery slope is a snow-covered slope that is covered in powder, or loose snow. Powdery slopes are often found in ski resorts, where skiers and snowboarders can ride down them. 
        xxxxxx A sunny day is a day with clear skies and bright sunshine. Sunny days are often associated with warm weather, and are a common sight in the summer. xxxxxx
        ======
        Context: {caption}
        Question: {question}
        LLM_Knowledge:
    """
    """Extract 2–3 background-knowledge facts about the scene to help reason toward an answer."""
    llm = get_llm_knowledge_base(temperature=0.7)
    return llm.invoke(prompt_template).content
