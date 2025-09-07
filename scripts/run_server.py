"""
Main entry point for running the FastAPI server.
Corresponds to `scripts/run_server.py` in the detailed design document.
"""
import argparse
import uvicorn

from llm_lab_core.api.main import app

def main():
    parser = argparse.ArgumentParser(description="Run the FastAPI server for the LLM NPC.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reloading for development.")
    
    # The design doc specifies --config, --security, --functions, but in this implementation,
    # those are hardcoded in app/api/main.py for simplicity. A real implementation
    # would pass these paths to the app factory.
    
    args = parser.parse_args()

    uvicorn.run(
        "llm_lab_core.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        # In a more advanced setup, you might have an app factory
        # and pass configurations here.
        # factory=True 
    )

if __name__ == "__main__":
    main()
