import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

def create_sentiment_analyzer_agent():
    """
    Creates a sentiment analyzer agent that can analyze the emotional tone of text.
    
    Uses the LiteLlm wrapper to leverage LLM capabilities for sentiment analysis.
    
    Returns:
        Agent: A configured sentiment analyzer agent
    """
    # Define available tools (if any external tools are needed)
    tools = [{"googleSearch": {}}]
    
    # Define the LLM with appropriate parameters
    llm = LiteLlm(
        model="gemini/gemini-1.5-flash", 
        tools=tools, 
        reasoning_effort="high", 
        thinking={"type": "enabled", "budget_tokens": 1024},
        api_key=os.environ.get("GOOGLE_API_KEY")
    )
    
    # Create the Sentiment Analyzer agent
    sentiment_agent = Agent(
        name="sentiment_analyzer_agent",
        description="Analyzes the sentiment and emotional tone of text content.",
        model=llm,
        instruction=(
            "You are a Sentiment Analysis specialist. Your purpose is to analyze the emotional tone of text, particularly Reddit posts and comments.\n\n"
            "When provided with text content:\n"
            "1. Analyze the sentiment as positive, negative, or neutral.\n"
            "2. Provide a confidence score (0-100%) for your sentiment classification.\n"
            "3. Identify key emotional markers or trigger words that influenced your analysis.\n"
            "4. Summarize the overall emotional tone in 1-2 sentences.\n\n"
            "Format your response as a structured analysis with clear sections:\n"
            "- Overall Sentiment: [positive/negative/neutral]\n"
            "- Confidence: [percentage]\n"
            "- Key Markers: [list of emotional terms or phrases]\n"
            "- Analysis: [brief explanation of the emotional tone]\n\n"
            "Keep your analysis objective, evidence-based, and sensitive to nuance in the text."
        )
    )
    
    return sentiment_agent

# Expose root_agent for ADK discovery
root_agent = create_sentiment_analyzer_agent()
