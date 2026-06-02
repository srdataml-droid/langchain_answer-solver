from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun#, GoogleSearchRun, PythonREPLToolRun, TerminalToolRun, RequestsGetToolRun, RequestsPostToolRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime
import json


search = DuckDuckGoSearchRun()
search_tool = Tool(
    name = "search",
    func = search.run,
    description = "search the web for information.",

)

#for the top_k_results = 1 , doc_content_chars_max = 100, the top result can return more than 1 result but we set one so it returns one and the characters contained inn the reult would not be exceeding 100 characters. This is to make sure that the result is concise and relevant to the query.
api_wrapper = WikipediaAPIWrapper(top_k_results = 1 ,doc_content_chars_max = 1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

def safe_wikipedia_run(query: str) -> str:
    """Safely cleans agent inputs before sending them to Wikipedia."""
    if isinstance(query, str) and ("{" in query or "}" in query):
        try:
            # Fix Python single quotes for valid JSON parsing
            clean_json_str = query.replace("'", '"')
            parsed_data = json.loads(clean_json_str)
            if isinstance(parsed_data, dict) and "query" in parsed_data:
                return wiki_tool.run(parsed_data["query"])
        except Exception:
            pass  # Fall back to raw string if parsing fails
    return wiki_tool.run(str(query)) 