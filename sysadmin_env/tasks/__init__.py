from sysadmin_env.models import DiagnosticTrigger
from sysadmin_env.models import TaskScenarioDefinition
from sysadmin_env.models import TaskScenarioState
from sysadmin_env.tasks import disk_full
from sysadmin_env.tasks import hpc_gpu_ecc
from sysadmin_env.tasks import hpc_munge
from sysadmin_env.tasks import hpc_nfs_stale
from sysadmin_env.tasks import hpc_ood_apache
from sysadmin_env.tasks import hpc_outage
from sysadmin_env.tasks import hpc_pid_stale
from sysadmin_env.tasks import network_broken
from sysadmin_env.tasks import nginx_crash


TASK_MODULES = {
    nginx_crash.TASK_ID: nginx_crash,
    disk_full.TASK_ID: disk_full,
    network_broken.TASK_ID: network_broken,
    hpc_outage.TASK_ID: hpc_outage,
    hpc_munge.TASK_ID: hpc_munge,
    hpc_pid_stale.TASK_ID: hpc_pid_stale,
    hpc_gpu_ecc.TASK_ID: hpc_gpu_ecc,
    hpc_nfs_stale.TASK_ID: hpc_nfs_stale,
    hpc_ood_apache.TASK_ID: hpc_ood_apache,
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
