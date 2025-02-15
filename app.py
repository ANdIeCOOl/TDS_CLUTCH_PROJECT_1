


from fastapi import FastAPI, HTTPException
import requests
import subprocess
import json
import os

app = FastAPI()

OPENAI_API_KEY = "your_openai_api_key"
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
MODEL = "gpt-4"
CODE_FILE = "generated_script.py"
RESPONSE_FORMAT = "json"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "dependencies": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["code"]
}

async def generate_code(task: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": "You are an experienced Python coder. Generate only Python code along with any required dependencies in a structured format, no explanations."},
                     {"role": "user", "content": task}],
        "response_format": RESPONSE_FORMAT
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, json=payload)
    response_json = response.json()
    return json.loads(response_json['choices'][0]['message']['content'])


def save_code(code: str):
    with open(CODE_FILE, "w", encoding="utf-8") as f:
        f.write(code)


def execute_code():
    try:
        result = subprocess.run(["python", CODE_FILE], capture_output=True, text=True, timeout=10)
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "Execution timed out"


@app.post("/generate-and-run")
async def generate_and_run(task: str):
    response = await generate_code(task)
    code = response.get("code", "")
    dependencies = response.get("dependencies", [])
    
    save_code(code)
    stdout, stderr = execute_code()
    
    if stderr:
        # Retry once with error feedback
        error_task = f"The previous code for task '{task}' gave this error: {stderr}. The code was: {code}. Fix it and regenerate the correct code along with dependencies."
        corrected_response = await generate_code(error_task)
        corrected_code = corrected_response.get("code", "")
        save_code(corrected_code)
        stdout, stderr = execute_code()
    
    return {"stdout": stdout, "stderr": stderr, "dependencies": dependencies}