@echo off
REM Start both embedding service and Streamlit app
REM This script starts the embedding service in the background and then starts Streamlit

echo Starting embedding service...
start "Embedding Service" cmd /k "python embedding_service.py"

REM Wait a moment for the service to start
timeout /t 3 /nobreak >nul

echo Starting Streamlit app...
streamlit run app.py

REM Note: The embedding service window will remain open
REM Close it manually when you're done

