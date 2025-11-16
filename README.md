# FastAPI Hello World

A single-file FastAPI application that exposes a `/` endpoint returning a JSON hello world message.

## Quickstart

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/ to see the greeting or visit http://127.0.0.1:8000/docs for interactive Swagger UI.
