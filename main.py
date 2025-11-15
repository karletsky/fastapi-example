from fastapi import FastAPI

app = FastAPI(title="Hello World API", version="0.1.0")


@app.get("/", summary="Hello World", tags=["greetings"])
def read_root() -> dict[str, str]:
    """Return a friendly greeting."""
    return {"message": "Hello, world!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
