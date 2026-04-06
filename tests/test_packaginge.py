import ast
import tomllib
from pathlib import Path


def test_openenv_manifest_declares_runtime_resources_and_tasks():
    manifest = Path("openenv.yaml").read_text()

    assert 'name: sysadmin-env' in manifest
    assert 'entry_point: inference.py' in manifest
    assert 'server_entry_point: server.app:app' in manifest
    assert 'reset_endpoint: /reset' in manifest
    assert 'step_endpoint: /step' in manifest
    assert 'state_endpoint: /state' in manifest
    assert 'websocket_endpoint: /ws' in manifest
    assert 'healthcheck_endpoint: /health' in manifest
    assert 'tasks_endpoint: /tasks' in manifest
    assert 'max_runtime_minutes: 20' in manifest
    assert 'id: nginx_crash' in manifest
    assert 'id: disk_full' in manifest
    assert 'id: network_broken' in manifest


def test_requirements_are_pinned_for_core_dependencies():
    lines = [line.strip() for line in Path("requirements.txt").read_text().splitlines() if line.strip()]
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    runtime_dependencies = pyproject["project"]["dependencies"]

    assert lines == runtime_dependencies
    assert 'fastapi==0.115.12' in lines
    assert 'uvicorn==0.34.2' in lines
    assert 'pydantic==2.11.1' in lines
    assert 'websockets==15.0.1' in lines
    assert 'httpx==0.28.1' in lines
    assert 'openenv-core>=0.2.0' in lines
    assert 'openai==2.11.0' in lines
    assert all('==' in line or line == 'openenv-core>=0.2.0' for line in lines)


def test_pyproject_declares_uv_friendly_project_metadata():
    pyproject = Path("pyproject.toml").read_text()
    pyproject_data = tomllib.loads(pyproject)
    python_version = Path(".python-version").read_text().strip()

    assert 'name = "sysadmin-env"' in pyproject
    assert 'requires-python = ">=3.11,<3.14"' in pyproject
    assert '"fastapi==0.115.12"' in pyproject
    assert '"openenv-core>=0.2.0"' in pyproject
    assert '"openai==2.11.0"' in pyproject
    assert 'server = "server.app:main"' in pyproject
    assert 'sysadmin-env-agent' not in pyproject
    assert pyproject_data["project"]["scripts"] == {"server": "server.app:main"}
    assert pyproject_data["tool"]["setuptools"]["py-modules"] == ["client", "inference", "models"]
    assert python_version == "3.11"


def test_openenv_push_required_structure_files_exist():
    assert Path("__init__.py").exists()
    assert Path("client.py").exists()
    assert Path("models.py").exists()
    assert Path("server/__init__.py").exists()
    assert Path("server/app.py").exists()
    assert Path("server/Dockerfile").exists()


def test_server_app_defines_main_function_and_standard_main_guard():
    server_app = Path("server/app.py").read_text()
    module = ast.parse(server_app)

    main_function = next(
        (node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"),
        None,
    )
    main_guard = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"
        ),
        None,
    )

    assert main_function is not None
    assert main_guard is not None
    assert any(
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "main"
        for statement in main_guard.body
    )


def test_dockerignore_excludes_local_state_and_secrets():
    dockerignore = Path(".dockerignore").read_text()

    assert '.env' in dockerignore
    assert '!.env.example' in dockerignore
    assert '__pycache__/' in dockerignore
    assert 'assets/runtime/' in dockerignore
    assert 'tests/' in dockerignore
    assert 'markdownstochat/' in dockerignore


def test_env_example_defaults_to_all_tasks():
    env_example = Path(".env.example").read_text()

    assert 'HF_TOKEN=""' in env_example
    assert 'MODEL_NAME="gpt-5.4"' in env_example
    assert 'SYSADMIN_ENV_TASK_ID=""' in env_example


def test_deployment_files_reference_hugging_face_and_openenv_workflow():
    dockerfile = Path("Dockerfile").read_text()
    server_dockerfile = Path("server/Dockerfile").read_text()
    readme = Path("README.md").read_text()
    client = Path("client.py").read_text()
    models = Path("models.py").read_text()
    server_app = Path("server/app.py").read_text()

    assert 'FROM python:3.11-slim' in dockerfile
    assert 'FROM python:3.11-slim' in server_dockerfile
    assert 'pyproject.toml' in dockerfile
    assert 'pyproject.toml' in server_dockerfile
    assert 'uv.lock' in dockerfile
    assert 'uv.lock' in server_dockerfile
    assert 'uv sync --locked --no-dev --no-install-project' in dockerfile
    assert 'uv sync --locked --no-dev --no-install-project' in server_dockerfile
    assert 'uv sync --locked --no-dev --no-editable' in dockerfile
    assert 'uv sync --locked --no-dev --no-editable' in server_dockerfile
    assert 'requirements.txt' not in dockerfile
    assert 'requirements.txt' not in server_dockerfile
    assert 'client.py' in dockerfile
    assert 'models.py' in dockerfile
    assert 'client.py' in server_dockerfile
    assert 'models.py' in server_dockerfile
    assert 'CMD ["uv", "run", "server"' in dockerfile
    assert 'CMD ["uv", "run", "server"' in server_dockerfile
    assert 'EXPOSE 8000' in dockerfile
    assert 'EXPOSE 8000' in server_dockerfile
    assert 'sdk: docker' in readme
    assert 'Hugging Face' in readme
    assert 'openenv' in readme
    assert 'openenv init' in readme
    assert 'openenv validate' in readme
    assert 'openenv push' in readme
    assert 'uv run server' in readme
    assert 'pyproject.toml' in readme
    assert 'uv.lock' in readme
    assert 'server/app.py' in readme
    assert 'server/Dockerfile' in readme
    assert '__init__.py' in readme
    assert 'client.py' in readme
    assert 'models.py' in readme
    assert 'server.app:app' in readme
    assert 'Dockerfile' in readme
    assert 'python -m pip install .' in readme
    assert 'docker build -t sysadmin-env .' in readme
    assert 'docker run --rm -p 18000:8000 sysadmin-env' in readme
    assert 'curl http://127.0.0.1:18000/health' in readme
    assert 'API_BASE_URL' in readme
    assert 'MODEL_NAME' in readme
    assert 'HF_TOKEN' in readme
    assert '/reset' in readme
    assert '/home/cyclops' not in readme
    assert 'mamba activate' not in readme
    assert 'legacy' not in readme.lower()
    assert 'compatib' not in readme.lower()
    assert 'sysadmin_env.server:app' not in readme
    assert 'from inference import main' in client
    assert 'from sysadmin_env.models import Action' in models
    assert 'from sysadmin_env.server import app' in server_app
    assert 'uvicorn.run(' in server_app
    assert 'if __name__ == "__main__":' in server_app
