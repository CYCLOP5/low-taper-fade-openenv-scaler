"""Modal deployment for sysadmin-env.

Deploy:
    modal serve modal_app.py          # live-reload dev
    modal deploy modal_app.py         # permanent URL

The permanent URL will be:
    https://<your-modal-workspace>--sysadmin-env-serve.modal.run

Before deploying, create a Modal secret named "sysadmin-env-secrets":
    modal secret create sysadmin-env-secrets OPENENV_API_KEY=<your-token>

If the secret is omitted the server starts without auth (backward-compatible).
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Image: Debian slim + system sandbox tools + project installed as a package
# ---------------------------------------------------------------------------

_EXCLUDE = [
    ".git/**",
    "env/**",
    ".venv/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "runs/**",
    "outputs/**",
    "docs/assets/*.png",
    "*.egg-info/**",
]

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install(
        "bubblewrap",
        "proot",
        "fuse-overlayfs",
        "procps",
        "iputils-ping",
        "findutils",
        "curl",
        "ca-certificates",
    )
    # python slim only puts python3 under /usr/local/bin; task stubs use
    # #!/usr/bin/env python3 so we need a stable /usr/bin/python3 as well.
    .run_commands(
        "[ -x /usr/local/bin/python3 ] && [ ! -e /usr/bin/python3 ]"
        " && ln -sf /usr/local/bin/python3 /usr/bin/python3 || true"
    )
    .add_local_dir(".", remote_path="/app", ignore=_EXCLUDE, copy=True)
    .workdir("/app")
    .run_commands(
        "pip install --no-cache-dir --upgrade pip setuptools wheel",
        "pip install --no-cache-dir .",
    )
)

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app = modal.App("sysadmin-env", image=image)

# Try to attach the secret if it exists; if the user hasn't created it yet
# the server starts without auth (OPENENV_API_KEY will simply be unset).
try:
    _secrets = [modal.Secret.from_name("sysadmin-env-secrets")]
except Exception:
    _secrets = []


@app.function(
    min_containers=1,
    timeout=3600,
    secrets=_secrets,
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def serve() -> object:
    import os

    os.chdir("/app")
    # EpisodeManager is instantiated at import time (module-level create_app()
    # call in sysadmin_env/server.py).  Changing directory first ensures
    # Path.cwd()/"assets" resolves inside the image.
    from sysadmin_env.server import app as fastapi_app  # noqa: PLC0415

    return fastapi_app
