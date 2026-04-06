import asyncio
import json
from pathlib import Path

import inference as inference_module


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    def __init__(self, *, get_payload: dict | None = None, post_status_code: int = 200):
        self.get_payload = get_payload or {}
        self.post_status_code = post_status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def get(self, _url: str):
        return FakeResponse(200, self.get_payload)

    async def post(self, _url: str, headers: dict, json: dict):
        return FakeResponse(self.post_status_code, {})


class FakeWebSocket:
    def __init__(self):
        self.sent_messages = []
        self.received_messages = [
            json.dumps({
                "type": "episode_started",
                "task": {
                    "task_id": "nginx_crash",
                    "difficulty": "easy",
                    "description": "nginx crashed with stale pid and config syntax error",
                    "max_steps": 40,
                    "time_limit": 300.0,
                },
            }),
            json.dumps({
                "type": "observation",
                "task_id": "nginx_crash",
                "observation": {
                    "stdout": "nginx started",
                    "stderr": "",
                    "exit_code": 0,
                    "working_directory": "/",
                    "execution_time": 0.01,
                    "reward": 1.0,
                    "done": True,
                    "step_number": 1,
                    "max_steps": 40,
                },
            }),
        ]

    async def send(self, message: str) -> None:
        self.sent_messages.append(json.loads(message))

    async def recv(self) -> str:
        return self.received_messages.pop(0)


class FakeConnect:
    def __init__(self, websocket: FakeWebSocket):
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


def _config() -> inference_module.AgentConfig:
    return inference_module.AgentConfig(
        server_url="ws://127.0.0.1:8000/ws",
        healthcheck_url="http://127.0.0.1:8000/health",
        tasks_url="http://127.0.0.1:8000/tasks",
        model_api_url="https://example.invalid",
        model_name="test-model",
        reasoning_effort=None,
        api_key=None,
        api_timeout=1.0,
        episode_timeout=5.0,
        task_id=None,
    )


def test_load_task_sequence_uses_explicit_task_id_without_http():
    config = inference_module.AgentConfig(
        server_url="ws://127.0.0.1:8000/ws",
        healthcheck_url="http://127.0.0.1:8000/health",
        tasks_url="http://127.0.0.1:8000/tasks",
        model_api_url="https://example.invalid",
        model_name="test-model",
        reasoning_effort=None,
        api_key=None,
        api_timeout=1.0,
        episode_timeout=5.0,
        task_id="disk_full",
    )

    result = asyncio.run(inference_module.load_task_sequence(config))

    assert result == ["disk_full"]


def test_load_task_sequence_reads_tasks_endpoint(monkeypatch):
    monkeypatch.setattr(
        inference_module.httpx,
        "AsyncClient",
        lambda timeout: FakeAsyncClient(get_payload={
            "tasks": [
                {"task_id": "nginx_crash"},
                {"task_id": "disk_full"},
                {"task_id": "network_broken"},
            ],
        }),
    )

    result = asyncio.run(inference_module.load_task_sequence(_config()))

    assert result == ["nginx_crash", "disk_full", "network_broken"]


def test_load_config_reads_dotenv_file(monkeypatch, tmp_path: Path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "OPENAI_API_KEY=dotenv_key\n"
        "OPENAI_MODEL=dotenv_model\n"
        "OPENAI_REASONING_EFFORT=low\n"
        "SYSADMIN_ENV_TASK_ID=disk_full\n"
    )

    for key in [
        "OPENAI_API_KEY",
        "API_KEY",
        "OPENAI_MODEL",
        "MODEL_NAME",
        "OPENAI_REASONING_EFFORT",
        "SYSADMIN_ENV_TASK_ID",
    ]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("SYSADMIN_ENV_DOTENV_PATH", str(dotenv_path))

    config = inference_module.load_config()

    assert config.api_key == "dotenv_key"
    assert config.model_name == "dotenv_model"
    assert config.reasoning_effort == "low"
    assert config.task_id == "disk_full"


def test_build_model_request_payload_uses_responses_api_reasoning_effort():
    config = inference_module.AgentConfig(
        server_url="ws://127.0.0.1:8000/ws",
        healthcheck_url="http://127.0.0.1:8000/health",
        tasks_url="http://127.0.0.1:8000/tasks",
        model_api_url="https://api.openai.com/v1/responses",
        model_name="gpt-5.4-nano",
        reasoning_effort="low",
        api_key="test-key",
        api_timeout=1.0,
        episode_timeout=5.0,
        task_id=None,
    )

    payload = inference_module._build_model_request_payload(
        config,
        {"task_id": "nginx_crash"},
        None,
        [],
    )

    assert payload["model"] == "gpt-5.4-nano"
    assert payload["reasoning"] == {"effort": "low"}
    assert "input" in payload


def test_request_model_action_returns_none_on_rate_limit(monkeypatch, capsys):
    monkeypatch.setattr(
        inference_module.httpx,
        "AsyncClient",
        lambda timeout: FakeAsyncClient(post_status_code=429),
    )

    result = asyncio.run(
        inference_module.request_model_action(
            _config(),
            {"task_id": "nginx_crash"},
            None,
            [],
        )
    )

    assert result is None
    assert "step api 429" in capsys.readouterr().out


def test_run_episode_sends_action_and_emits_step_tags(monkeypatch, capsys):
    websocket = FakeWebSocket()

    async def fake_choose_action(config, task, observation, history):
        return inference_module.ModelDecision(
            command="echo ready",
            reasoning="fallback heuristic",
            source="fallback",
        )

    monkeypatch.setattr(inference_module, "choose_action", fake_choose_action)
    monkeypatch.setattr(
        inference_module.websockets,
        "connect",
        lambda url, open_timeout: FakeConnect(websocket),
    )

    asyncio.run(inference_module.run_episode(_config(), "nginx_crash"))

    output = capsys.readouterr().out
    assert "[step]" in output
    assert "episode_started" in output
    assert "echo ready" in output
    assert websocket.sent_messages == [{"command": "echo ready", "reasoning": "fallback heuristic"}]


def test_run_emits_start_and_end_tags_for_each_episode(monkeypatch, capsys):
    async def fake_verify_server(config):
        return None

    async def fake_load_task_sequence(config):
        return ["nginx_crash", "disk_full"]

    async def fake_run_episode(config, task_id):
        print(f"[step] {{\"task_id\": \"{task_id}\"}}")

    monkeypatch.setattr(inference_module, "verify_server", fake_verify_server)
    monkeypatch.setattr(inference_module, "load_task_sequence", fake_load_task_sequence)
    monkeypatch.setattr(inference_module, "run_episode", fake_run_episode)

    exit_code = asyncio.run(inference_module.run())

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.count("[start]") == 2
    assert output.count("[end]") == 2
    assert "nginx_crash" in output
    assert "disk_full" in output


def test_heuristic_action_produces_task_specific_safe_commands():
    nginx_decision = inference_module.heuristic_action({"task_id": "nginx_crash"}, None, [])
    disk_decision = inference_module.heuristic_action({"task_id": "disk_full"}, None, [])
    network_decision = inference_module.heuristic_action({"task_id": "network_broken"}, None, [])

    assert nginx_decision.command == "cat /var/log/nginx/error.log"
    assert disk_decision.command == "df -h /mnt/data"
    assert network_decision.command == "ip route show"
    assert nginx_decision.source == "fallback"
