from sysadmin_env.models import DiagnosticTrigger
from sysadmin_env.models import TaskScenarioDefinition
from sysadmin_env.models import TaskScenarioState
from sysadmin_env.tasks import disk_full
from sysadmin_env.tasks import network_broken
from sysadmin_env.tasks import nginx_crash


TASK_MODULES = {
    nginx_crash.TASK_ID: nginx_crash,
    disk_full.TASK_ID: disk_full,
    network_broken.TASK_ID: network_broken,
}


def build_task_registry(base_root: str) -> dict[str, TaskScenarioDefinition]:
    return {
        task_id: module.build_definition(f"{base_root}/{task_id}")
        for task_id, module in TASK_MODULES.items()
    }


def get_task_module(task_id: str):
    return TASK_MODULES[task_id]


__all__ = [
    "DiagnosticTrigger",
    "TaskScenarioDefinition",
    "TaskScenarioState",
    "TASK_MODULES",
    "build_task_registry",
    "get_task_module",
]
