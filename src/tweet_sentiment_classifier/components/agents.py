import os
import json
from typing import List, Dict, Any, Optional
# pyrefly: ignore [missing-import]
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
import feedparser
import requests
from tweet_sentiment_classifier.config.env_config import get_settings

settings = get_settings()

class SearchAgent:
    def __init__(self):
        self.ddg_search = DuckDuckGoSearchRun()
        # Fallback RSS feeds
        self.fallback_feeds = [
            "https://news.ycombinator.com/rss",
            "http://feeds.bbci.co.uk/news/rss.xml"
        ]

    def search(self, query: str) -> Dict[str, Any]:
        results = []
        fallback_used = False
        try:
            # Attempt DuckDuckGo search
            ddg_result = self.ddg_search.run(query)
            if ddg_result and len(ddg_result.strip()) > 50:
                # split roughly into sentences or paragraphs to simulate multiple texts
                results = [text.strip() for text in ddg_result.split('.') if len(text.strip()) > 10]
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")

        # Fallback if not enough data
        if len(results) < 3:
            fallback_used = True
            print("Not enough data from DDG, falling back to RSS feeds...")
            for feed_url in self.fallback_feeds:
                try:
                    # Added timeout optimization
                    response = requests.get(feed_url, timeout=3.0)
                    feed = feedparser.parse(response.content)
                    for entry in feed.entries[:5]: # Take top 5 from each feed
                        results.append(entry.title + ": " + getattr(entry, 'description', ''))
                except Exception as e:
                    print(f"RSS fallback failed for {feed_url}: {e}")

        return {"results": results, "fallback_used": fallback_used}

class ReportAgent:
    def __init__(self):
        # Uses GROQ_API_KEY from settings
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant", 
            temperature=0.2, 
            api_key=settings.GROQ_API_KEY
        )
        
        template = """
        You are an analytical assistant. You are provided with some texts obtained from a search query on the internet, and the sentiment prediction for each text from our ML backend.
        
        Search Query: {query}
        
        Data:
        {data}
        
        Task: Create a JSON report summarizing the findings. Your output MUST be ONLY valid JSON, with the following structure:
        {{
            "prediction_explanation": "A single string paragraph explaining why the texts received their sentiments based on their content",
            "sentiment_report": "A single string paragraph summarizing the overall sentiment of the search results",
            "insights": "A single string containing 2-3 key insights from the texts, separated by newlines",
            "confidence_summary": "A single string summarizing the prediction confidences",
            "keyword_extraction": ["keyword1", "keyword2", "keyword3"]
        }}
        """
        self.prompt = PromptTemplate(template=template, input_variables=["query", "data"])
        self.chain = self.prompt | self.llm
        
    def generate_report(self, query: str, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        data_str = json.dumps(predictions, indent=2)
        try:
            response = self.chain.invoke({"query": query, "data": data_str})
            content = response.content
            
            # Attempt to parse json from response
            # Strip out markdown formatting if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            return json.loads(content)
        except Exception as e:
            print(f"Failed to generate report from LLM: {e}")
            # Fallback
            return {
                "prediction_explanation": "Failed to generate explanation.",
                "sentiment_report": "Failed to generate report.",
                "insights": "N/A",
                "confidence_summary": "N/A",
                "keyword_extraction": []
            }
