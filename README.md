# AI Study Assistant

This project contains a backend, frontend, data layer, and tests for an AI-powered study assistant.

## Project Structure

- `backend/` - API server and business logic
- `frontend/` - UI application
- `data/` - datasets, prompts, and supporting files
- `tests/` - test suite
- `.env.example` - environment configuration template
- `requirements.txt` - Python dependencies

## Getting Started

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .\.venv\Scripts\activate    # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy environment variables:
   ```bash
   copy .env.example .env
   ```
4. Start the backend:
   ```bash
   uvicorn backend.main:app --reload
   ```

## Notes

This is the initial scaffold for the project. Add your app-specific code under the relevant folders as development continues.
