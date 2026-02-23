import re
from typing import Dict
from loguru import logger

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Options: ["None", "Wiki", "Search", "Reflection"]
def post_process(raw_inputs:Dict[str,str], output:str, post_method:str):
    if post_method == None or post_method == "None":
        return output
    elif post_method == "Wiki":
        return wiki(raw_inputs, output)
    elif post_method == "Search":
        return search(raw_inputs, output)
    elif post_method == "Reflection":
        return reflection(raw_inputs, output)
    else:
        raise ValueError(f"Invalid post-processing method: {post_method}")


def wiki(raw_inputs:Dict[str,str], output:str):
    """
    Extract information from Wikipedia to post-process the output.
    Args:
        output: str: The output from the LLM.
    Returns:
        Any: str: The post-processed output.
    """
    pattern = r'```keyword.*```'
    match = re.search(pattern, output, re.DOTALL|re.MULTILINE)
    if match:
        keywords = match.group(0).lstrip("```keyword\n").rstrip("\n```")
        logger.info(f"keywords: {keywords}")
        try:
            keywords = eval(keywords)
        except Exception:
            logger.warning("Wiki keyword evaluation failed, skipping")
            keywords = []
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
        for keyword in keywords:
            if type(keyword) == str or type(keyword) == dict:
                output += f"\n{wikipedia.run(keyword)}"
    return output

def search(raw_inputs:Dict[str,str], output:str):
    """
    Search for information to post-process the output.
    Args:
        output: str: The output from the LLM.
    Returns:
        Any: str: The post-processed output.
    """
    #TODO Search for information to post-process the output
    #Maybe bocha or brave search

    return output

def reflection(raw_inputs:Dict[str,str], output:str):
    """
    Reflect on the output to post-process the output.
    Args:
        output: str: The output from the LLM.
    Returns:
        Any: str: The post-processed output.
    """
    return output
