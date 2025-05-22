"""
Unified Agent Dashboard
======================

This Streamlit application provides a comprehensive interface for interacting with all
available ADK agents in the project. It integrates multiple agent capabilities into a single
interface with tabs for different functionalities.

Features:
- Reddit content browsing
- Sentiment analysis
- Text summarization
- Text-to-speech conversion
- Coordinated workflows between agents

Requirements:
------------
- All standalone A2A agents must be running
  * Speaker Agent: python -m agents.speaker (port 8003)
  * Sentiment Analyzer: python -m agents.sentiment_analyzer (port 8004)
- Alternatively, use adk api_server for non-A2A integrated functionality
- Streamlit and related packages installed

Usage:
------
1. Ensure all required agent servers are running
2. Run this dashboard: `streamlit run apps/unified_dashboard.py`
3. Use the tabs to navigate between different agent functionalities
"""

import streamlit as st
import requests
import json
import uuid
import time
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO

# Set up basic logging for the app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Unified Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants for API endpoints
REDDIT_SCOUT_URL = "http://0.0.0.0:8000/run"  # Using ADK API server
SPEAKER_URL = "http://127.0.0.1:8003/run"     # A2A endpoint
SENTIMENT_ANALYZER_URL = "http://127.0.0.1:8004/run"  # A2A endpoint
API_BASE_URL = "http://0.0.0.0:8000"  # ADK API server for regular agent interactions

# Define supported subreddits
SUBREDDITS = ["gamedev", "unity3d", "unrealengine", "programming", "indiegamedev"]

# Initialize session state variables
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user-{uuid.uuid4()}"

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session-{int(time.time())}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "reddit_posts" not in st.session_state:
    st.session_state.reddit_posts = []

if "current_summary" not in st.session_state:
    st.session_state.current_summary = ""

if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = {}

if "audio_file" not in st.session_state:
    st.session_state.audio_file = None

# Colors for sentiment visualization
SENTIMENT_COLORS = {
    "positive": "#4CAF50",  # Green
    "neutral": "#FFC107",   # Amber
    "negative": "#F44336",  # Red
    "unknown": "#9E9E9E"    # Gray
}

# Helper functions for agent interaction

def create_adk_session():
    """Create a new session with the ADK API server."""
    session_id = f"session-{int(time.time())}"
    try:
        response = requests.post(
            f"{API_BASE_URL}/apps/agents.coordinator/users/{st.session_state.user_id}/sessions/{session_id}",
            headers={"Content-Type": "application/json"},
            data=json.dumps({})
        )
        
        if response.status_code == 200:
            st.session_state.session_id = session_id
            st.session_state.messages = []
            return True
        else:
            st.error(f"Failed to create session: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error creating session: {e}")
        return False

def send_to_reddit_scout(subreddit):
    """Send a request to the Reddit Scout agent via the coordinator."""
    try:
        with st.spinner(f"Fetching posts from r/{subreddit}..."):
            # Using coordinator agent to access Reddit Scout
            response = requests.post(
                REDDIT_SCOUT_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "app_name": "agents.coordinator",  # Use full module path
                    "user_id": st.session_state.user_id,
                    "session_id": st.session_state.session_id,
                    "new_message": {
                        "role": "user",
                        "parts": [{"text": f"Show me the latest posts from r/{subreddit}"}]
                    }
                }
            )
            
            if response.status_code != 200:
                st.error(f"Error: {response.text}")
                return None
                
            events = response.json()
            
            # Extract the final response
            posts_text = None
            for event in events:
                if event.get("content", {}).get("role") == "model":
                    posts_text = event.get("content", {}).get("parts", [{}])[0].get("text")
            
            # Debug output to see the response
            st.write("Raw response from Reddit agent:")
            st.code(posts_text, language="text")
            
            # Parse the posts from the response text
            if posts_text:
                lines = posts_text.split('\n')
                posts = []
                for line in lines:
                    line = line.strip()
                    if line.startswith('- '):
                        posts.append(line[2:])  # Remove the bullet point
                    elif len(line) > 0 and not line.startswith("Top posts from") and not line.startswith("Here are"):
                        # Catch posts that might not have bullet points
                        posts.append(line)
                
                # Debug info
                st.write(f"Found {len(posts)} posts")
                
                # Only store if we actually found posts
                if posts:
                    st.session_state.reddit_posts = posts
                    return posts
                else:
                    st.warning("No posts found in response. Please try again.")
            else:
                st.warning("No text response from Reddit agent.")
            
            return None
    except Exception as e:
        st.error(f"Error fetching Reddit posts: {e}")
        logger.exception("Error in Reddit Scout request")
        return None

def send_to_summarizer(text):
    """Send text to the Summarizer agent via the coordinator."""
    try:
        with st.spinner("Generating summary..."):
            # Using coordinator agent to access Summarizer
            response = requests.post(
                REDDIT_SCOUT_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "app_name": "agents.coordinator",  # Use full module path
                    "user_id": st.session_state.user_id,
                    "session_id": st.session_state.session_id,
                    "new_message": {
                        "role": "user",
                        "parts": [{"text": f"Summarize this: {text}"}]
                    }
                }
            )
            
            if response.status_code != 200:
                st.error(f"Error: {response.text}")
                return None
                
            events = response.json()
            
            # Extract the final response
            summary = None
            for event in events:
                if event.get("content", {}).get("role") == "model":
                    summary = event.get("content", {}).get("parts", [{}])[0].get("text")
            
            st.session_state.current_summary = summary
            return summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        logger.exception("Error in Summarizer request")
        return None

def send_to_sentiment_analyzer(text):
    """Send text to the Sentiment Analyzer agent via A2A protocol."""
    try:
        with st.spinner("Analyzing sentiment..."):
            payload = {
                "message": text,
                "context": {
                    "user_id": st.session_state.user_id
                },
                "session_id": st.session_state.session_id
            }
            
            response = requests.post(
                SENTIMENT_ANALYZER_URL,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                st.error(f"Error: {response.text}")
                return None
            
            response_data = response.json()
            
            analysis_message = response_data.get("message", "")
            sentiment_data = response_data.get("data", {}).get("sentiment_analysis", {})
            
            st.session_state.current_analysis = {
                "message": analysis_message,
                "data": sentiment_data
            }
            
            return sentiment_data
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        logger.exception("Error in Sentiment Analyzer request")
        return None

def send_to_speaker(text):
    """Send text to the Speaker agent via A2A protocol."""
    try:
        with st.spinner("Converting text to speech..."):
            payload = {
                "message": text,
                "context": {
                    "user_id": st.session_state.user_id
                },
                "session_id": st.session_state.session_id
            }
            
            response = requests.post(
                SPEAKER_URL,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                st.error(f"Error: {response.text}")
                return None
            
            response_data = response.json()
            
            message = response_data.get("message", "")
            audio_url = response_data.get("data", {}).get("audio_url")
            
            if audio_url:
                if audio_url.startswith("file://"):
                    audio_path = audio_url.replace("file://", "")
                else:
                    audio_path = audio_url
                    
                if os.path.exists(audio_path):
                    st.session_state.audio_file = audio_path
                    return audio_path
            
            return None
    except Exception as e:
        st.error(f"Error converting to speech: {e}")
        logger.exception("Error in Speaker request")
        return None

def create_sentiment_gauge(sentiment, confidence):
    """Create a matplotlib gauge chart for sentiment visualization."""
    # Extract confidence as a float
    if isinstance(confidence, str) and '%' in confidence:
        confidence_value = float(confidence.strip('%')) / 100
    else:
        try:
            confidence_value = float(confidence) / 100 if float(confidence) > 1 else float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.5
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 1), facecolor='none')
    
    # Determine color based on sentiment
    color = SENTIMENT_COLORS.get(sentiment.lower(), SENTIMENT_COLORS["unknown"])
    
    # Create a horizontal bar
    ax.barh([0], [1], color='#EEEEEE', height=0.3)
    ax.barh([0], [confidence_value], color=color, height=0.3)
    
    # Add a marker for the confidence value
    ax.scatter(confidence_value, 0, color='black', s=80, zorder=5)
    
    # Remove axes and set limits
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    
    return fig

def display_sentiment_emoji(sentiment):
    """Return an emoji based on sentiment."""
    if sentiment == "positive":
        return "😊"
    elif sentiment == "negative":
        return "😞"
    elif sentiment == "neutral":
        return "😐"
    else:
        return "❓"

# UI Components

# Sidebar for general controls
with st.sidebar:
    st.title("🤖 Agent Dashboard")
    st.subheader("Session Information")
    
    # Session management for ADK API server agents
    if st.button("🔄 Create New ADK Session"):
        if create_adk_session():
            st.success(f"Created new session: {st.session_state.session_id}")
        
    st.info(f"User ID: {st.session_state.user_id}")
    st.info(f"Session ID: {st.session_state.session_id}")
    
    st.divider()
    
    # Information about required services
    st.subheader("Required Services")
    st.caption("Ensure these services are running:")
    st.caption("• ADK API Server (`adk api_server`)")
    st.caption("• Speaker Agent (`python -m agents.speaker`)")
    st.caption("• Sentiment Analyzer (`python -m agents.sentiment_analyzer`)")

# Main content with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Reddit Content", "Summarization", "Sentiment Analysis", "Text-to-Speech"])

# Tab 1: Reddit Content
with tab1:
    st.header("Reddit Content Explorer")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Subreddit selection and fetch
        selected_subreddit = st.selectbox("Select Subreddit", SUBREDDITS)
        
        if st.button("Fetch Posts", key="fetch_reddit"):
            send_to_reddit_scout(selected_subreddit)
    
    with col2:
        # Display fetched posts
        if st.session_state.reddit_posts:
            st.subheader(f"Latest Posts from r/{selected_subreddit}")
            
            for i, post in enumerate(st.session_state.reddit_posts):
                post_container = st.container(border=True)
                with post_container:
                    st.write(post)
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("Summarize", key=f"summarize_{i}"):
                            summary = send_to_summarizer(post)
                            if summary:
                                st.session_state.current_summary = summary
                                st.rerun()
                    
                    with col_b:
                        if st.button("Analyze Sentiment", key=f"sentiment_{i}"):
                            sentiment = send_to_sentiment_analyzer(post)
                            if sentiment:
                                st.rerun()
                    
                    with col_c:
                        if st.button("Convert to Speech", key=f"speak_{i}"):
                            audio_path = send_to_speaker(post)
                            if audio_path:
                                st.rerun()
            
            # Analysis on all posts together
            st.divider()
            if st.button("Analyze All Posts Together"):
                combined_text = "\n".join(st.session_state.reddit_posts)
                send_to_summarizer(combined_text)
                send_to_sentiment_analyzer(combined_text)
                st.rerun()
        else:
            st.info("Select a subreddit and click 'Fetch Posts' to begin")

# Tab 2: Summarization
with tab2:
    st.header("Text Summarization")
    
    # Input area for text to summarize
    with st.expander("Enter custom text to summarize", expanded=not bool(st.session_state.current_summary)):
        custom_text = st.text_area("Text to summarize", height=200)
        
        if st.button("Generate Summary", key="custom_summary"):
            if custom_text:
                send_to_summarizer(custom_text)
                st.rerun()
            else:
                st.warning("Please enter some text to summarize")
    
    # Display the summary if available
    if st.session_state.current_summary:
        st.subheader("Summary")
        summary_container = st.container(border=True)
        with summary_container:
            st.write(st.session_state.current_summary)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Analyze Sentiment", key="summary_sentiment"):
                    send_to_sentiment_analyzer(st.session_state.current_summary)
                    st.rerun()
            
            with col_b:
                if st.button("Convert to Speech", key="summary_speak"):
                    send_to_speaker(st.session_state.current_summary)
                    st.rerun()

# Tab 3: Sentiment Analysis
with tab3:
    st.header("Sentiment Analysis")
    
    # Input area for text to analyze
    with st.expander("Enter custom text to analyze", expanded=not bool(st.session_state.current_analysis)):
        analysis_text = st.text_area("Text to analyze", height=200)
        
        if st.button("Analyze Sentiment", key="custom_sentiment"):
            if analysis_text:
                send_to_sentiment_analyzer(analysis_text)
                st.rerun()
            else:
                st.warning("Please enter some text to analyze")
    
    # Display sentiment analysis if available
    if st.session_state.current_analysis and "data" in st.session_state.current_analysis:
        sentiment_data = st.session_state.current_analysis["data"]
        sentiment = sentiment_data.get("sentiment", "unknown")
        confidence = sentiment_data.get("confidence", "0%")
        markers = sentiment_data.get("key_markers", [])
        analysis = sentiment_data.get("analysis", "No analysis available")
        
        st.subheader("Sentiment Analysis Results")
        
        # Visual representation of sentiment
        cols = st.columns([1, 4])
        with cols[0]:
            st.markdown(f"# {display_sentiment_emoji(sentiment)}")
            st.markdown(f"### {sentiment.capitalize()}")
        
        with cols[1]:
            st.pyplot(create_sentiment_gauge(sentiment, confidence))
            st.write(f"**Confidence:** {confidence}")
        
        # Key markers and analysis
        st.subheader("Key Markers")
        if markers:
            for marker in markers:
                st.markdown(f"- {marker}")
        else:
            st.info("No key markers identified")
        
        st.subheader("Analysis")
        st.write(analysis)
        
        # Actions
        if st.button("Convert Analysis to Speech", key="analysis_speak"):
            full_text = f"Sentiment Analysis: {sentiment} with {confidence} confidence. {analysis}"
            send_to_speaker(full_text)
            st.rerun()

# Tab 4: Text-to-Speech
with tab4:
    st.header("Text-to-Speech Conversion")
    
    # Input area for text to speak
    with st.expander("Enter text to convert to speech", expanded=not bool(st.session_state.audio_file)):
        speech_text = st.text_area("Text to speak", height=200)
        
        if st.button("Convert to Speech", key="custom_speech"):
            if speech_text:
                send_to_speaker(speech_text)
                st.rerun()
            else:
                st.warning("Please enter some text to convert to speech")
    
    # Display audio player if available
    if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
        st.subheader("Generated Audio")
        try:
            with open(st.session_state.audio_file, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            st.audio(audio_bytes)
            
            # Show file path
            st.caption(f"Audio file path: {st.session_state.audio_file}")
            
            # Option to analyze the text that was converted
            spoken_text = speech_text if 'speech_text' in locals() and speech_text else "Unknown text"
            if st.button("Analyze Sentiment of this Text", key="speech_sentiment"):
                send_to_sentiment_analyzer(spoken_text)
                st.rerun()
        except Exception as e:
            st.error(f"Error loading audio file: {e}")

# Footer
st.divider()
st.caption("ADK Made Simple - Unified Agent Dashboard")
