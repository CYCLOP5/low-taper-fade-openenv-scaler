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
    def __init__(self, *, get_payload: dict | None = None):
        self.get_payload = get_payload or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def get(self, _url: str):
        return FakeResponse(200, self.get_payload)


class FakeOpenAIResponse:
    def __init__(self, output_text: str = '{"command": "echo ready", "reasoning": "safe step"}', status: str = "completed"):
        self.output_text = output_text
        self.status = status
        self.incomplete_details = None


class FakeResponsesApi:
    def __init__(self, *, response: FakeOpenAIResponse | None = None, exc: Exception | None = None):
        self.response = response or FakeOpenAIResponse()
        self.exc = exc
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.response


class FakeOpenAIClient:
    def __init__(self, *, response: FakeOpenAIResponse | None = None, exc: Exception | None = None):
        self.responses = FakeResponsesApi(response=response, exc=exc)
        self.closed = False

    def close(self):
        self.closed = True


class FakeStatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


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
        model_api_url="https://api.openai.com/v1",
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
        model_api_url="https://api.openai.com/v1",
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


def test_load_config_reads_required_env_names(monkeypatch, tmp_path: Path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "HF_TOKEN=dotenv_key\n"
        "MODEL_NAME=dotenv_model\n"
        "API_BASE_URL=https://example.test/v1\n"
        "OPENAI_REASONING_EFFORT=medium\n"
        "SYSADMIN_ENV_TASK_ID=disk_full\n"
    )

    for key in [
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "API_KEY",
        "API_BASE_URL",
        "OPENAI_BASE_URL",
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
    assert config.model_api_url == "https://example.test/v1"
    assert config.reasoning_effort == "medium"
    assert config.task_id == "disk_full"


def test_load_config_prefers_current_working_directory_dotenv(monkeypatch, tmp_path: Path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "MODEL_NAME=cwd_model\n"
        "API_BASE_URL=https://cwd.example/v1\n"
    )

    for key in [
        "SYSADMIN_ENV_DOTENV_PATH",
        "MODEL_NAME",
        "OPENAI_MODEL",
        "API_BASE_URL",
        "OPENAI_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.chdir(tmp_path)

    config = inference_module.load_config()

    assert config.model_name == "cwd_model"
    assert config.model_api_url == "https://cwd.example/v1"


def test_build_model_request_payload_uses_openai_responses_shape():
    config = inference_module.AgentConfig(
        server_url="ws://127.0.0.1:8000/ws",
        healthcheck_url="http://127.0.0.1:8000/health",
        tasks_url="http://127.0.0.1:8000/tasks",
        model_api_url="https://api.openai.com/v1",
        model_name="gpt-5.4-nano",
        reasoning_effort="medium",
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
    assert payload["reasoning"] == {"effort": "medium"}
    assert "instructions" in payload
    assert "input" in payload


def test_build_model_request_payload_uses_generic_network_playbook_guidance():
    payload = inference_module._build_model_request_payload(
        _config(),
        {"task_id": "network_broken"},
        None,
        [],
    )

    user_payload = json.loads(payload["input"])
    playbook = user_payload["playbook"]

    assert "repair_targets" not in playbook
    assert playbook["supported_repairs"][0] == "write the repaired default route into /etc/network/routes/default"
    assert playbook["avoid"][0] == "do not guess host-specific gateways or dns servers without evidence from the task"


def test_request_model_action_returns_none_on_rate_limit(monkeypatch, capsys):
    monkeypatch.setattr(
        inference_module,
        "_create_openai_client",
        lambda config: FakeOpenAIClient(exc=FakeStatusError(429)),
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
    assert "step api 429" in capsys.readouterr().err


def test_request_model_action_parses_json_output(monkeypatch):
    monkeypatch.setattr(
        inference_module,
        "_create_openai_client",
        lambda config: FakeOpenAIClient(response=FakeOpenAIResponse('{"command": "echo ready", "reasoning": "repair now"}')),
    )

    result = asyncio.run(
        inference_module.request_model_action(
            inference_module.AgentConfig(
                server_url="ws://127.0.0.1:8000/ws",
                healthcheck_url="http://127.0.0.1:8000/health",
                tasks_url="http://127.0.0.1:8000/tasks",
                model_api_url="https://api.openai.com/v1",
                model_name="test-model",
                reasoning_effort=None,
                api_key="test-key",
                api_timeout=1.0,
                episode_timeout=5.0,
                task_id=None,
            ),
            {"task_id": "nginx_crash"},
            None,
            [],
        )
    )

    assert result is not None
    assert result.command == "echo ready"
    assert result.reasoning == "repair now"
    assert result.source == "model"


def test_choose_action_uses_network_guardrail_after_diagnosis(monkeypatch):
    async def fake_request_model_action(config, task, observation, history):
        return inference_module.ModelDecision(
            command="ip route replace default via 172.17.0.1 dev eth0",
            reasoning="common container repair",
            source="model",
        )

    config = _config()
    config.api_key = "test-key"
    monkeypatch.setattr(inference_module, "request_model_action", fake_request_model_action)

    decision = asyncio.run(
        inference_module.choose_action(
            config,
            {"task_id": "network_broken"},
            None,
            [
                {"action": "ip route show", "observation": {"reward": 0.07}},
                {"action": "ip -br addr", "observation": {"reward": 0.05}},
                {"action": "cat /etc/resolv.conf", "observation": {"reward": 0.05}},
            ],
        )
    )

    assert decision.source == "fallback"
    assert decision.command == "printf 'nameserver 1.1.1.1\n' > /etc/resolv.conf"


def test_choose_action_keeps_supported_network_repair_from_model(monkeypatch):
    async def fake_request_model_action(config, task, observation, history):
        return inference_module.ModelDecision(
            command="ip route add default via 10.0.2.2",
            reasoning="repair the route using the supported stub",
            source="model",
        )

    config = _config()
    config.api_key = "test-key"
    monkeypatch.setattr(inference_module, "request_model_action", fake_request_model_action)

    decision = asyncio.run(
        inference_module.choose_action(
            config,
            {"task_id": "network_broken"},
            None,
            [
                {"action": "ip route show", "observation": {"reward": 0.07}},
                {"action": "ip addr", "observation": {"reward": 0.05}},
                {"action": "cat /etc/resolv.conf", "observation": {"reward": 0.05}},
            ],
        )
    )

    assert decision.source == "model"
    assert decision.command == "ip route add default via 10.0.2.2"


def test_choose_action_network_guardrail_advances_to_route_repair_after_dns(monkeypatch):
    async def fake_request_model_action(config, task, observation, history):
        return inference_module.ModelDecision(
            command="ip route replace default via 172.17.0.1 dev eth0",
            reasoning="common container repair",
            source="model",
        )

    config = _config()
    config.api_key = "test-key"
    monkeypatch.setattr(inference_module, "request_model_action", fake_request_model_action)

    decision = asyncio.run(
        inference_module.choose_action(
            config,
            {"task_id": "network_broken"},
            None,
            [
                {"action": "ip route show", "observation": {"reward": 0.07}},
                {"action": "ip addr", "observation": {"reward": 0.05}},
                {"action": "cat /etc/resolv.conf", "observation": {"reward": 0.05}},
                {"action": "printf 'nameserver 1.1.1.1\n' > /etc/resolv.conf", "observation": {"reward": 0.19}},
            ],
        )
    )

    assert decision.source == "fallback"
    assert decision.command == "printf 'default via 10.0.2.2 dev eth0\n' > /etc/network/routes/default"


def test_choose_action_network_guardrail_does_not_accept_failed_dns_guess(monkeypatch):
    async def fake_request_model_action(config, task, observation, history):
        return inference_module.ModelDecision(
            command="ip route replace default via 172.17.0.1 dev eth0",
            reasoning="common container repair",
            source="model",
        )

    config = _config()
    config.api_key = "test-key"
    monkeypatch.setattr(inference_module, "request_model_action", fake_request_model_action)

    decision = asyncio.run(
        inference_module.choose_action(
            config,
            {"task_id": "network_broken"},
            None,
            [
                {"action": "ip route show", "observation": {"reward": 0.07}},
                {"action": "ip addr", "observation": {"reward": 0.05}},
                {"action": "cat /etc/resolv.conf", "observation": {"reward": 0.05}},
                {
                    "action": "sh -c 'printf \"nameserver 1.1.1.1\\nnameserver 8.8.8.8\\n\" > /etc/resolv.conf'",
                    "observation": {"reward": -0.01},
                },
            ],
        )
    )

    assert decision.source == "fallback"
    assert decision.command == "printf 'nameserver 1.1.1.1\n' > /etc/resolv.conf"


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

    summary = asyncio.run(inference_module.run_episode(_config(), "nginx_crash"))

    output = capsys.readouterr().out
    assert "[STEP] step=1 action=echo ready reward=1.00 done=true error=null" in output
    assert summary.success is True
    assert summary.steps == 1
    assert websocket.sent_messages == [{"command": "echo ready", "reasoning": "fallback heuristic"}]


def test_run_emits_start_and_end_tags_for_each_episode(monkeypatch, capsys):
    async def fake_verify_server(config):
        return None

    async def fake_load_task_sequence(config):
        return ["nginx_crash", "disk_full"]

    async def fake_run_episode(config, task_id):
        return inference_module.EpisodeSummary(
            task_id=task_id,
            success=True,
            steps=1,
            score=0.99,
            rewards=[1.0],
        )

    monkeypatch.setattr(inference_module, "verify_server", fake_verify_server)
    monkeypatch.setattr(inference_module, "load_task_sequence", fake_load_task_sequence)
    monkeypatch.setattr(inference_module, "run_episode", fake_run_episode)

    exit_code = asyncio.run(inference_module.run())

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.count("[START]") == 2
    assert output.count("[END]") == 2
    assert "[START] task=nginx_crash env=sysadmin-env model=" in output
    assert "[START] task=disk_full env=sysadmin-env model=" in output
    assert "[END] success=true steps=1 score=0.99 rewards=1.00" in output


def test_log_helpers_support_legacy_json_mode(monkeypatch, capsys):
    monkeypatch.setenv("SYSADMIN_ENV_LOG_FORMAT", "json")

    inference_module.log_start(task="network_broken", env="sysadmin-env", model="test-model")
    inference_module.log_step(step=2, action="ip route show", reward=0.07, done=False, error=None)
    inference_module.log_end(success=True, steps=2, score=1.0, rewards=[0.07, 0.93])

    output = capsys.readouterr().out
    assert "[START] {\"task\": \"network_broken\"" in output
    assert "[STEP] {\"step\": 2, \"action\": \"ip route show\"" in output
    assert "[END] {\"success\": true, \"steps\": 2, \"score\": 1.0" in output


def test_log_end_flat_format_includes_score(capsys):
    inference_module.log_end(success=True, steps=3, score=0.98, rewards=[0.35, 0.24, 0.39])

    output = capsys.readouterr().out.strip()
    assert output == "[END] success=true steps=3 score=0.98 rewards=0.35,0.24,0.39"


def test_normalize_reported_score_maps_to_open_interval():
    assert inference_module._normalize_reported_score(-5.0) == 0.01
    assert inference_module._normalize_reported_score(0.0) == 0.01
    assert inference_module._normalize_reported_score(1.0) == 0.99
    assert inference_module._normalize_reported_score(5.0) == 0.99


def test_normalize_openai_base_url_strips_responses_suffix():
    assert inference_module._normalize_openai_base_url("https://api.openai.com/v1/responses") == "https://api.openai.com/v1"
    assert inference_module._normalize_openai_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"


def test_heuristic_action_produces_task_specific_safe_commands():
    nginx_decision = inference_module.heuristic_action({"task_id": "nginx_crash"}, None, [])
    disk_decision = inference_module.heuristic_action({"task_id": "disk_full"}, None, [])
    network_decision = inference_module.heuristic_action({"task_id": "network_broken"}, None, [])

    assert nginx_decision.command == "cat /var/log/nginx/error.log"
    assert disk_decision.command == "df -h /mnt/data"
    assert network_decision.command == "ip route show"
    assert nginx_decision.source == "fallback"
