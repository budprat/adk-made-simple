"""
Task Manager for the Sentiment Analyzer Agent.
Handles sentiment analysis requests using the ADK Runner.
"""

import os
import logging
import uuid
import re
from typing import Dict, Any, Optional

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types as adk_types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define app name for the runner
A2A_APP_NAME = "sentiment_analyzer_a2a_app"

class SentimentTaskManager:
    """Task Manager for the Sentiment Analyzer Agent in A2A mode."""
    
    def __init__(self, agent: Agent):
        """Initialize with an Agent instance and set up ADK Runner."""
        logger.info(f"Initializing SentimentTaskManager for agent: {agent.name}")
        self.agent = agent
        
        # Initialize ADK services
        self.session_service = InMemorySessionService()
        self.artifact_service = InMemoryArtifactService()
        
        # Create the runner
        self.runner = Runner(
            agent=self.agent,
            app_name=A2A_APP_NAME,
            session_service=self.session_service,
            artifact_service=self.artifact_service
        )
        logger.info(f"ADK Runner initialized for app '{self.runner.app_name}'")

    async def process_task(self, message: str, context: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an A2A task request by running the agent.
        
        Args:
            message: The text message to analyze for sentiment.
            context: Additional context data.
            session_id: Session identifier (generated if None).
            
        Returns:
            Response dict with message, status, and structured sentiment data.
        """
        # Get user_id from context or use default
        user_id = context.get("user_id", "default_a2a_user")
        
        # Create or get session
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session_id: {session_id}")
            
        session = self.session_service.get_session(app_name=A2A_APP_NAME, user_id=user_id, session_id=session_id)
        if not session:
            session = self.session_service.create_session(app_name=A2A_APP_NAME, user_id=user_id, session_id=session_id, state={})
            logger.info(f"Created new session: {session_id}")
        
        # Create user message
        request_content = adk_types.Content(role="user", parts=[adk_types.Part(text=message)])
        
        try:
            # Run the agent
            events_async = self.runner.run_async(
                user_id=user_id, 
                session_id=session_id, 
                new_message=request_content
            )
            
            # Process response
            final_message = "(No response generated)"
            raw_events = []
            sentiment_data = {
                "sentiment": "unknown",
                "confidence": "0%",
                "key_markers": [],
                "analysis": "No analysis available"
            }

            # Process events
            async for event in events_async:
                raw_events.append(event.model_dump(exclude_none=True))
                
                # Extract from the final response
                if event.is_final_response() and event.content and event.content.role == "model":
                    if event.content.parts and event.content.parts[0].text:
                        final_message = event.content.parts[0].text
                        logger.info(f"Received sentiment analysis response")
                        
                        # Extract structured sentiment data
                        sentiment_data = self._extract_sentiment_data(final_message)
            
            # Return formatted response
            return {
                "message": final_message, 
                "status": "success",
                "data": {
                    "sentiment_analysis": sentiment_data,
                    "raw_events": raw_events[-3:]  # Include last few events for debugging
                }
            }

        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return {
                "message": f"Error processing your request: {str(e)}",
                "status": "error",
                "data": {"error_type": type(e).__name__}
            }
    
    def _extract_sentiment_data(self, text: str) -> Dict[str, Any]:
        """
        Extract structured sentiment data from the agent's response text.
        
        Args:
            text: The agent's response text
            
        Returns:
            Dictionary containing structured sentiment data
        """
        # Default values
        result = {
            "sentiment": "unknown",
            "confidence": "0%",
            "key_markers": [],
            "analysis": "No analysis available"
        }
        
        # Try to extract sentiment
        sentiment_match = re.search(r"Overall Sentiment: *(positive|negative|neutral)", text, re.IGNORECASE)
        if sentiment_match:
            result["sentiment"] = sentiment_match.group(1).lower()
        
        # Try to extract confidence
        confidence_match = re.search(r"Confidence: *(\d+%|\d+)", text, re.IGNORECASE)
        if confidence_match:
            result["confidence"] = confidence_match.group(1)
            if not result["confidence"].endswith("%"):
                result["confidence"] += "%"
        
        # Try to extract key markers - grab anything between Key Markers: and the next section
        markers_match = re.search(r"Key Markers: *(.*?)(?=\n\n|\n-|$)", text, re.IGNORECASE | re.DOTALL)
        if markers_match:
            # Split by commas, newlines, or bullet points and clean up
            markers_text = markers_match.group(1).strip()
            markers = []
            
            # Handle bullet point lists
            if "-" in markers_text:
                for line in markers_text.split("\n"):
                    if "-" in line:
                        marker = line.split("-", 1)[1].strip()
                        if marker:
                            markers.append(marker)
            # Handle comma-separated lists
            elif "," in markers_text:
                markers = [m.strip() for m in markers_text.split(",") if m.strip()]
            # Handle single item
            elif markers_text:
                markers = [markers_text]
                
            result["key_markers"] = markers
        
        # Try to extract analysis
        analysis_match = re.search(r"Analysis: *(.*?)(?=\n\n|$)", text, re.IGNORECASE | re.DOTALL)
        if analysis_match:
            result["analysis"] = analysis_match.group(1).strip()
        
        return result
