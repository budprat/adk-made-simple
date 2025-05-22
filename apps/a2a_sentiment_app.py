"""
A2A Sentiment Analyzer Chat Application
======================================

This Streamlit application provides a chat interface for interacting with the
standalone A2A Sentiment Analyzer Agent. It allows users to send text and receive
detailed sentiment analysis.

Requirements:
------------
- Standalone Sentiment Analyzer Agent running (e.g., `python -m agents.sentiment_analyzer`) on localhost:8004
- Streamlit and related packages installed

Usage:
------
1. Start the Sentiment Analyzer Agent: `python -m agents.sentiment_analyzer`
2. Run this Streamlit app: `streamlit run apps/a2a_sentiment_app.py`
3. Start chatting with the Sentiment Analyzer Agent

Architecture:
------------
- Session Management: Uses Streamlit session state to manage user_id and session_id implicitly.
- Message Handling: Sends user messages to the A2A /run endpoint and processes responses.
- Results Display: Shows formatted sentiment analysis results with visual indicators.
"""
import streamlit as st
import requests
import json
import uuid
import time
import logging
import matplotlib.pyplot as plt
import numpy as np

# Set up basic logging for the app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="A2A Sentiment Analyzer",
    page_icon="😊",  # Emotion icon
    layout="centered"
)

# Constants
API_BASE_URL = "http://127.0.0.1:8004"  # Sentiment analyzer port

# Initialize session state variables
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user-{uuid.uuid4()}"  # Persistent user ID

if "session_id" not in st.session_state:
    st.session_state.session_id = f"conv-{uuid.uuid4()}"  # Unique conversation ID

if "messages" not in st.session_state:
    st.session_state.messages = []

# Colors for sentiment visualization
SENTIMENT_COLORS = {
    "positive": "#4CAF50",  # Green
    "neutral": "#FFC107",   # Amber
    "negative": "#F44336",  # Red
    "unknown": "#9E9E9E"    # Gray
}

def send_message(message):
    """
    Send a message to the sentiment analyzer agent and process the response.
    """
    if not st.session_state.session_id:
        st.error("No active conversation. Start typing to begin.")
        return False

    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": message})

    # Construct A2A payload
    payload = {
        "message": message,
        "context": {
            "user_id": st.session_state.user_id
        },
        "session_id": st.session_state.session_id
    }
    
    # Send POST request with spinner for UI feedback
    try:
        with st.spinner("Analyzing sentiment..."):
            response = requests.post(
                f"{API_BASE_URL}/run",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                json=payload,
                timeout=30
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Parse A2A response
            response_data = response.json()
            
            assistant_message = response_data.get("message", "(No analysis received)")
            # Extract sentiment data from the 'data' dictionary
            sentiment_data = response_data.get("data", {}).get("sentiment_analysis", {})
            
            # Add assistant response to chat history
            st.session_state.messages.append(
                {
                    "role": "assistant", 
                    "content": assistant_message, 
                    "sentiment_data": sentiment_data
                }
            )
            
            return True

    except requests.exceptions.RequestException as e:
        st.error(f"Network or HTTP error: {e}")
        logger.error(f"RequestException: {e}", exc_info=True)
        st.session_state.messages.append({"role": "assistant", "content": f"Error: Could not connect to agent. {e}"})
        return False
    except json.JSONDecodeError as e:
        st.error("Failed to decode JSON response from agent.")
        logger.error(f"JSONDecodeError: {e}", exc_info=True)
        st.session_state.messages.append({"role": "assistant", "content": "Error: Invalid response format from agent."}) 
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logger.error(f"Unexpected error in send_message: {e}", exc_info=True)
        st.session_state.messages.append({"role": "assistant", "content": f"Error: An unexpected error occurred. {e}"}) 
        return False

def create_sentiment_gauge(sentiment, confidence):
    """Create a simple matplotlib gauge chart for sentiment visualization."""
    # Extract confidence as a float from the percentage string
    if isinstance(confidence, str) and '%' in confidence:
        confidence_value = float(confidence.strip('%')) / 100
    else:
        try:
            confidence_value = float(confidence) / 100 if float(confidence) > 1 else float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.5  # Default to 50% if unparseable
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(4, 0.6), facecolor='none')
    
    # Determine color based on sentiment
    color = SENTIMENT_COLORS.get(sentiment.lower(), SENTIMENT_COLORS["unknown"])
    
    # Create a horizontal bar
    ax.barh([0], [1], color='#EEEEEE', height=0.3)
    ax.barh([0], [confidence_value], color=color, height=0.3)
    
    # Add a marker for the confidence value
    ax.scatter(confidence_value, 0, color='black', s=50, zorder=5)
    
    # Remove axes and set limits
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    
    return fig

# UI Components
st.title("😊 Sentiment Analyzer")

# Sidebar for session management
with st.sidebar:
    st.header("Conversation Control")
    
    if st.button("🧹 New Conversation"):
        st.session_state.session_id = f"conv-{uuid.uuid4()}"
        st.session_state.messages = []
        st.success("Started new conversation.")
        st.rerun()
    
    if st.session_state.session_id:
        st.info(f"User ID: {st.session_state.user_id}")
        st.caption(f"Conversation ID: {st.session_state.session_id}")
    else:
        st.info("Start typing to analyze sentiment.")

    st.divider()
    st.caption("This app interacts directly with the Sentiment Analyzer via A2A.")
    st.caption("Make sure the agent is running on port 8004.")

# Chat interface
st.subheader("Text Sentiment Analysis")

# Display messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        with st.chat_message("assistant"):
            # If we have sentiment data, display it in a formatted way
            if "sentiment_data" in msg and isinstance(msg["sentiment_data"], dict):
                sentiment_data = msg["sentiment_data"]
                sentiment = sentiment_data.get("sentiment", "unknown")
                confidence = sentiment_data.get("confidence", "0%")
                markers = sentiment_data.get("key_markers", [])
                analysis = sentiment_data.get("analysis", "No analysis available")
                
                # Display sentiment in a visually appealing way
                col1, col2 = st.columns([1, 4])
                
                # Emoji in the first column
                with col1:
                    if sentiment == "positive":
                        st.markdown("# 😊")
                    elif sentiment == "negative":
                        st.markdown("# 😞")
                    elif sentiment == "neutral":
                        st.markdown("# 😐")
                    else:
                        st.markdown("# ❓")
                
                # Sentiment details in the second column
                with col2:
                    st.markdown(f"### {sentiment.capitalize()}")
                    st.markdown(f"**Confidence:** {confidence}")
                
                # Display the gauge
                st.pyplot(create_sentiment_gauge(sentiment, confidence))
                
                # Display key markers in an expandable section
                if markers:
                    with st.expander("Key Markers"):
                        for marker in markers:
                            st.markdown(f"- {marker}")
                
                # Display the analysis
                st.markdown(f"**Analysis:** {analysis}")
                
                # Include the full message in a collapsible section
                with st.expander("Full Message"):
                    st.write(msg["content"])
            else:
                # Just display the regular message if no sentiment data
                st.write(msg["content"])

# Input for new messages
user_input = st.chat_input("Enter text to analyze sentiment...")
if user_input:
    send_message(user_input)
    st.rerun()
