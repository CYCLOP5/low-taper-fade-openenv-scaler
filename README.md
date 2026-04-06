---
title: sysadmin env
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---




## run the server

```bash
python -m uvicorn sysadmin_env.server:app --host 0.0.0.0 --port 8000
```

## run the agent


```bash
source /home/cyclops/.bashrc
mamba activate metascaler
python inference.py
```

## use a .env file


```dotenv
OPENAI_API_KEY="your_key_here"
OPENAI_MODEL="gpt-5.4-pro?"
OPENAI_REASONING_EFFORT="low"
OPENAI_BASE_URL="https://api.openai.com/v1/responses"
SYSADMIN_ENV_SERVER_URL="ws://127.0.0.1:8000/ws"
SYSADMIN_ENV_HEALTHCHECK_URL="http://127.0.0.1:8000/health"
SYSADMIN_ENV_TASKS_URL="http://127.0.0.1:8000/tasks"
SYSADMIN_ENV_TASK_ID="nginx_crash"
```


then run

```bash
python inference.py
```

shell environment var will override values from [`.env`](.env).

if `OPENAI_API_KEY` is not set the system falls back to the built in heuristic agent.

## hugging face deployment

this repository is prepared for a Hugging Face docker space and the runtime entrypoint is defined in [Dockerfile](Dockerfile).


```bash
python -m pytest -q
```
