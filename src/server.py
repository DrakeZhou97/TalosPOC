import os

import uvicorn
from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import LangGraphAGUIAgent
from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver

from src.main import create_talos_agent

talos_agent = create_talos_agent(checkpointer=MemorySaver())

app = FastAPI()

add_langgraph_fastapi_endpoint(app, LangGraphAGUIAgent(name="labAssistant", graph=talos_agent), "/agent")


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        # reload=True,
    )
