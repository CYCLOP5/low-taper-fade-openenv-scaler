from pathlib import Path


def test_openenv_manifest_declares_runtime_resources_and_tasks():
    manifest = Path("openenv.yaml").read_text()

    assert 'name: sysadmin-env' in manifest
    assert 'entry_point: inference.py' in manifest
    assert 'server_entry_point: sysadmin_env.server:app' in manifest
    assert 'websocket_endpoint: /ws' in manifest
    assert 'healthcheck_endpoint: /health' in manifest
    assert 'tasks_endpoint: /tasks' in manifest
    assert 'max_runtime_minutes: 20' in manifest
    assert 'id: nginx_crash' in manifest
    assert 'id: disk_full' in manifest
    assert 'id: network_broken' in manifest


def test_requirements_are_pinned_for_core_dependencies():
    lines = [line.strip() for line in Path("requirements.txt").read_text().splitlines() if line.strip()]

    assert 'fastapi==0.115.12' in lines
    assert 'uvicorn==0.34.2' in lines
    assert 'pydantic==2.11.1' in lines
    assert 'websockets==15.0.1' in lines
    assert 'httpx==0.28.1' in lines
    assert all('==' in line for line in lines)


def test_deployment_files_reference_hugging_face_and_uvicorn():
    dockerfile = Path("Dockerfile").read_text()
    readme = Path("README.md").read_text()

    assert 'FROM python:3.11-slim' in dockerfile
    assert 'uvicorn' in dockerfile
    assert 'EXPOSE 8000' in dockerfile
    assert 'sdk: docker' in readme
    assert 'Hugging Face' in readme
