"""
Standalone A2A Sentiment Analyzer Agent
=======================================

This module provides a standalone FastAPI server for the Sentiment Analyzer agent
using the A2A protocol. It can be run directly to start the agent as a service:

```
python -m agents.sentiment_analyzer
```

This will start the A2A server on port 8004 by default.
"""

import asyncio
import os
import uvicorn
from dotenv import load_dotenv
from agents.sentiment_analyzer.agent import create_sentiment_analyzer_agent
from agents.sentiment_analyzer.task_manager import SentimentTaskManager
from common.a2a_server import create_agent_server

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Default port (different from Speaker's 8003)
DEFAULT_PORT = 8004

async def main():
    """Start the Sentiment Analyzer agent as a standalone A2A server."""
    # Create the agent instance
    sentiment_agent = create_sentiment_analyzer_agent()
    
    # Create a task manager with the agent
    task_manager = SentimentTaskManager(sentiment_agent)
    
    # Create A2A server using the task manager
    app = create_agent_server(
        name="Sentiment Analyzer",
        description="Analyzes sentiment and emotional tone in text content",
        task_manager=task_manager
    )
    
    # Start the server
    port = int(os.environ.get("SENTIMENT_ANALYZER_PORT", DEFAULT_PORT))
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    print(f"Starting Sentiment Analyzer A2A server on port {port}")
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
