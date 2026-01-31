#!/bin/bash
# Start both embedding service and Streamlit app
# This script starts the embedding service in the background and then starts Streamlit

echo "Starting embedding service..."
python embedding_service.py &
EMBEDDING_PID=$!

# Wait a moment for the service to start
sleep 3

echo "Starting Streamlit app..."
streamlit run app.py

# Cleanup: kill embedding service when Streamlit exits
echo "Stopping embedding service..."
kill $EMBEDDING_PID 2>/dev/null

