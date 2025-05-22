#!/bin/bash
# Test script for the Sentiment Analyzer Agent (A2A mode)
# This script sends a test message to the A2A Sentiment Analyzer
# and checks if the response contains sentiment analysis data

# Ensure the directory exists
mkdir -p "$(dirname "$0")"

# Configuration
AGENT_URL="http://localhost:8004/run"
USER_ID="test-user-$(date +%s)"
SESSION_ID="test-session-$(date +%s)"

# Test message
TEST_MESSAGE="I'm feeling extremely happy and excited about the new features in this project. The code quality is excellent and everything works perfectly!"

# Create the request payload
REQUEST='{
  "message": "'"$TEST_MESSAGE"'",
  "context": {
    "user_id": "'"$USER_ID"'"
  },
  "session_id": "'"$SESSION_ID"'"
}'

echo "🧪 Testing Sentiment Analyzer Agent via A2A protocol..."
echo "🔍 Sending test message: '$TEST_MESSAGE'"
echo "📡 URL: $AGENT_URL"
echo "🔑 Session ID: $SESSION_ID"
echo "⏳ Sending request..."

# Send the request and capture the response
RESPONSE=$(curl -s -X POST "$AGENT_URL" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d "$REQUEST")

# Check if the request was successful
if [ $? -ne 0 ]; then
  echo "❌ Error: Failed to connect to the agent. Is it running at $AGENT_URL?"
  exit 1
fi

# Extract sentiment data
SENTIMENT=$(echo "$RESPONSE" | grep -o '"sentiment":"[^"]*"' | cut -d'"' -f4)
CONFIDENCE=$(echo "$RESPONSE" | grep -o '"confidence":"[^"]*"' | cut -d'"' -f4)

# Check if sentiment data exists
if [ -z "$SENTIMENT" ]; then
  echo "❌ Error: No sentiment data found in response."
  echo "📝 Response: $RESPONSE"
  exit 1
fi

echo "✅ Test successful!"
echo "🎯 Detected sentiment: $SENTIMENT"
echo "📊 Confidence: $CONFIDENCE"
echo "📝 Full response:"
echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"

exit 0
