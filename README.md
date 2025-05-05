# Expense Tracker

A smart expense tracking app with AI-powered categorization.

## Features
- Natural language expense entry
- Automatic categorization
- Category filtering
- Real-time total calculation

## Setup
1. Clone the repo
2. Create `.env` file with your OpenAI API key:
   ```text
   OPENAI_API_KEY=your_key_here
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   uvicorn main:app --reload
   ```
