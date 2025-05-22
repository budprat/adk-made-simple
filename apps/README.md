# ADK Streamlit Applications

This directory contains Streamlit applications demonstrating different ways to interact with the ADK agents in this project.

## Applications

### Agent-Specific UIs

1.  **`speaker_app.py` (ADK API Server Mode)**
    - **Purpose:** Interacts with the Speaker Agent *through* the main `adk api_server`.
    - **Requires:** `adk api_server` running on `http://localhost:8000`.
    - **Run:** `streamlit run apps/speaker_app.py`

2.  **`a2a_speaker_app.py` (Standalone A2A Mode)**
    - **Purpose:** Interacts *directly* with the Speaker Agent running as a standalone Agent-to-Agent (A2A) service.
    - **Requires:** Standalone Speaker Agent running (`python -m agents.speaker`) on `http://localhost:8003`.
    - **Run:** `streamlit run apps/a2a_speaker_app.py`

3.  **`a2a_sentiment_app.py` (Sentiment Analyzer)**
    - **Purpose:** Interacts with the Sentiment Analyzer Agent to analyze the emotional tone of text.
    - **Requires:** Standalone Sentiment Analyzer Agent running (`python -m agents.sentiment_analyzer`) on `http://localhost:8004`.
    - **Run:** `streamlit run apps/a2a_sentiment_app.py`

### Integrated Dashboard

4.  **`unified_dashboard.py` (Comprehensive UI)**
    - **Purpose:** Integrates all agent capabilities in a single tabbed interface.
    - **Requires:** 
      - `adk api_server` running on `http://localhost:8000` (for Reddit Scout and Summarizer)
      - Standalone Speaker Agent running on `http://localhost:8003`
      - Standalone Sentiment Analyzer Agent running on `http://localhost:8004`
    - **Features:**
      - Reddit Content tab: Browse posts from selected subreddits
      - Summarization tab: Generate concise summaries of text
      - Sentiment Analysis tab: Analyze the emotional tone of text with visualizations
      - Text-to-Speech tab: Convert text to audio with ElevenLabs TTS
    - **Run:** `streamlit run apps/unified_dashboard.py`

## Key Differences in Interaction

### ADK API Server vs. Standalone A2A

The primary differences lie in how requests are sent and responses are handled:

| Feature              | Applications via `adk api_server`                                    | Applications via Standalone A2A                                |
| :------------------- | :-------------------------------------------------------------------- | :--------------------------------------------------------------- |
| **Target URL**       | `http://localhost:8000/run`                                           | Agent-specific port (e.g., `:8003` or `:8004`)                  |
| **Session Handling** | Explicit: Must call `/apps/.../sessions` endpoint first (via UI button) | Implicit: `session_id` sent with each `/run` request             |
| **Request Payload**  | ADK structure (`app_name`, `user_id`, `session_id`, `new_message`)    | Simple A2A structure (`message`, `context`, `session_id`)        |
| **Response Format**  | Stream of ADK event JSON objects                                      | Single structured JSON (`AgentResponse` model)                   |
| **Response Parsing** | Iterates through events, looks for `model` role or `functionResponse` | Directly accesses fields like `message` and `data.audio_url`     |
| **Agent Definition** | Relies on `adk api_server` using agent definition files               | Relies on standalone `__main__.py` instantiating the `Agent`     |

### Request Payload Example (`speaker_app.py` -> `adk api_server`)

```json
{
  "app_name": "speaker",
  "user_id": "user-xxx",
  "session_id": "session-yyy",
  "new_message": {
    "role": "user",
    "parts": [{"text": "Say hello world"}]
  }
}
```

### Request Payload Example (`a2a_speaker_app.py` -> Standalone Agent)

```json
{
  "message": "Say hello world",
  "context": {
    "user_id": "user-xxx"
  },
  "session_id": "conv-zzz"
}
```

### Agent-Specific Apps vs. Unified Dashboard

The unified dashboard (`unified_dashboard.py`) integrates features from all agents:

| Feature                  | Agent-Specific Apps                                               | Unified Dashboard                                             |
| :----------------------- | :---------------------------------------------------------------- | :------------------------------------------------------------ |
| **Interface**            | Single purpose, focused on one agent's capabilities               | Tabbed interface with multiple agent capabilities              |
| **Session Management**   | Individual session per app                                        | Shared session across tabs                                     |
| **Data Handling**        | Isolated to single agent interaction                              | Cross-agent data flow (e.g., analyze sentiment of Reddit posts) |
| **Visualization**        | Basic representation of agent outputs                             | Enhanced visualizations (charts, gauges, styled containers)    |
| **Workflow**             | Linear interaction with one agent at a time                       | Integrated workflows combining multiple agent capabilities     |

### Response Handling

#### ADK API Server Mode (`speaker_app.py`, etc.)

The app needs to loop through multiple JSON events returned by the `adk api_server`. It looks for specific event structures, like a `functionResponse` from the `text_to_speech` tool, often extracting information from nested text fields (like log messages).

#### Standalone A2A Mode (`a2a_speaker_app.py`, `a2a_sentiment_app.py`, etc.)

The app receives a single JSON object. It directly accesses keys like `response['message']` and `response['data']['audio_url']` based on the expected `AgentResponse` structure defined in `common/a2a_server.py`.

These differences illustrate the distinct communication patterns when interacting with an ADK agent via the framework's `api_server` versus a direct A2A connection to a standalone agent service.
