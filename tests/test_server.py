import json
from pathlib import Path

from fastapi.testclient import TestClient

from sysadmin_env.server import create_app


class FakeSandbox:
    def __init__(self, lowerdir, *, timeout=30.0, isolate_network=True, overlay_base_dir=None):
        self.lowerdir = Path(lowerdir)
        self.timeout = timeout
        self.isolate_network = isolate_network
        self.overlay_base_dir = overlay_base_dir
        self.merged_root = self.lowerdir
        self.created = False
        self.destroyed = False
        self.commands = []

    def create(self):
        self.created = True

    async def execute_async(self, command: str):
        self.commands.append(command)
        if command == "sleep 999":
            return type("Result", (), {
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "execution_time": 1.5,
                "timed_out": True,
            })()
        return type("Result", (), {
            "stdout": f"ran {command}",
            "stderr": "",
            "exit_code": 0,
            "execution_time": 0.02,
            "timed_out": False,
        })()

    def destroy(self):
        self.destroyed = True


def _make_client(monkeypatch):
    import sysadmin_env.server as server_module

    monkeypatch.setattr(server_module, "Sandbox", FakeSandbox)
    app = create_app()
    return TestClient(app)


def test_health_endpoint(monkeypatch):
    client = _make_client(monkeypatch)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_tasks_endpoint_returns_registered_scenarios(monkeypatch):
    client = _make_client(monkeypatch)
    response = client.get("/tasks")
    assert response.status_code == 200
    payload = response.json()
    assert "tasks" in payload
    assert len(payload["tasks"]) == 3
    assert {task["task_id"] for task in payload["tasks"]} == {"nginx_crash", "disk_full", "network_broken"}


def test_http_reset_step_and_state_contract(monkeypatch):
    client = _make_client(monkeypatch)

    reset_response = client.post("/reset", json={"task_id": "nginx_crash"})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["state"]["task_id"] == "nginx_crash"
    assert reset_payload["state"]["step_count"] == 0
    assert reset_payload["observation"]["done"] is False

    step_response = client.post("/step", json={"action": {"command": "echo hello"}})
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert step_payload["observation"]["stdout"] == "ran echo hello"
    assert step_payload["state"]["step_count"] == 1

    state_response = client.get("/state")
    assert state_response.status_code == 200
    assert state_response.json()["step_count"] == 1


def test_http_step_requires_reset(monkeypatch):
    client = _make_client(monkeypatch)
    response = client.post("/step", json={"action": {"command": "echo hello"}})
    assert response.status_code == 409


def test_websocket_handles_valid_invalid_and_timeout_actions(monkeypatch):
    client = _make_client(monkeypatch)

    with client.websocket_connect("/ws?task_id=nginx_crash") as websocket:
        started = websocket.receive_json()
        assert started["type"] == "episode_started"
        assert started["task"]["task_id"] == "nginx_crash"

        websocket.send_text("not json")
        error_payload = websocket.receive_json()
        assert error_payload == {
            "type": "error",
            "code": "invalid_action",
            "message": "malformed action json",
        }

        websocket.send_text(json.dumps({"command": "echo hello"}))
        observation_payload = websocket.receive_json()
        assert observation_payload["type"] == "observation"
        assert observation_payload["task_id"] == "nginx_crash"
        observation = observation_payload["observation"]
        assert observation["stdout"] == "ran echo hello"
        assert observation["exit_code"] == 0
        assert observation["step_number"] == 1
        assert observation["max_steps"] == 40
        assert observation["done"] is False

        websocket.send_text(json.dumps({"command": "sleep 999"}))
        timed_out_payload = websocket.receive_json()
        timeout_observation = timed_out_payload["observation"]
        assert timeout_observation["exit_code"] == -1
        assert "command execution timed out" in timeout_observation["stderr"]
        assert timeout_observation["step_number"] == 2


def test_websocket_disconnect_cleans_up_sandbox(monkeypatch):
    import sysadmin_env.server as server_module

    created = []

    class TrackingSandbox(FakeSandbox):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(server_module, "Sandbox", TrackingSandbox)
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/ws?task_id=disk_full") as websocket:
        websocket.receive_json()

    assert created
    assert created[0].destroyed is True
