# Sentiment Analyzer Agent Test Scripts

This directory contains test scripts for the Sentiment Analyzer Agent.

## Scripts

- **`test_a2a_sentiment.sh`**: Tests the Sentiment Analyzer Agent running as a standalone A2A service. Sends a sample text to the agent and verifies that it returns valid sentiment analysis data.

## Usage

1. Ensure the Sentiment Analyzer Agent is running as a standalone A2A service:
   ```bash
   # From the project root
   python -m agents.sentiment_analyzer
   ```

2. Run the test script:
   ```bash
   # From the project root
   bash scripts/tests/sentiment/test_a2a_sentiment.sh
   ```

3. Expected Output:
   - The script will send a test message to the agent
   - It will verify the agent returns proper sentiment data (sentiment type and confidence)
   - If successful, it will display the full response
   - If unsuccessful, it will report the error

## Notes

- The test script expects the Sentiment Analyzer Agent to be running on port 8004
- The test uses a positive message by default, which should result in a "positive" sentiment classification
- The script performs basic validation on the response format to ensure the agent is working correctly
