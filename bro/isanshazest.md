# execution phases



## phase one, pydantic models and data contracts

goal is to define the exact data shapes for actions, observations, task metadata, and reward signals. these models are the contract between every other component. nothing else can be built correctly without them.

deliverables.
implement the action model in models.py with command string and optional reasoning string.
implement the observation model in models.py with stdout, stderr, exit code, working directory, execution time, reward, done flag, step number, max steps.
implement a task metadata model with fields for task id, difficulty, description, max steps, time limit, base filesystem path.
implement a reward signal model with health delta, knowledge delta, action penalty, total reward.

validation.
instantiate each model with sample data and confirm serialization to json.
confirm that invalid data raises pydantic validation errors.


---

## phase two, overlayfs state engine

goal is to build the filesystem layer manager that enables sub second state resets. this is the foundation that the sandbox depends on.

deliverables.
implement the overlayfs manager class in overlayfs.py.
method to create a new overlay stack given a lowerdir path, creating upperdir, workdir, and merged directories.
method to mount the overlay using the mount syscall or mount command with appropriate fallbacks for unprivileged contexts.
method to reset the overlay by deleting upperdir contents and recreating an empty upperdir.
method to unmount and clean up all directories.
handle the case where fuse overlayfs is needed instead of kernel overlayfs in unprivileged environments.

validation.
create an overlay stack over a test lowerdir.
write a file into the merged directory and confirm it appears in upperdir.
reset the overlay and confirm the written file is gone.
confirm the lowerdir is never modified.
measure reset latency and confirm sub second performance.


---

## phase three, bubblewrap sandbox manager

goal is to build the sandbox lifecycle manager that isolates agent commands. depends on phase two for filesystem state.

deliverables.
implement the sandbox manager class in sandbox.py.
method to create a sandbox by constructing the bubblewrap command line with namespace flags for mount, pid, network, and user isolation.
method to execute a command inside the sandbox capturing stdout, stderr, exit code, and wall clock execution time.
enforce a per command timeout to prevent the agent from hanging the system.
method to reset the sandbox state by calling the overlayfs reset method.
method to destroy the sandbox releasing all resources.

the bubblewrap invocation must.
bind mount the overlayfs merged directory as the sandbox root.
create new pid, mount, and user namespaces.
optionally create a new network namespace depending on the task requirements.
set the working directory to root inside the sandbox.
drop all capabilities.
run as an unprivileged user.

validation.
create a sandbox and execute echo hello, confirm stdout contains hello.
execute a file write command, reset the sandbox, confirm the file is gone.
execute a command that exceeds the timeout, confirm it is killed and returns an error.
confirm the sandbox cannot access the host filesystem outside the mounted overlay.


---

## phase four, task scenario engine

goal is to build the three task scenarios with their base filesystems, fault injection, and grading functions. depends on phase two and three.

### task one, easy, nginx crash with stale pid and config syntax error

base filesystem contents.
a minimal linux filesystem with nginx installed.
an nginx binary or stub that can be started and stopped.
an nginx configuration file with a deliberate syntax error on a specific line.
a stale pid file at the standard nginx pid location pointing to a nonexistent process.

grading function.
check if the stale pid file has been removed, partial credit.
check if the nginx config syntax error has been fixed, partial credit.
check if nginx is running and responding, full credit.

diagnostic knowledge rewards.
reading the nginx error log.
running nginx t to test configuration.
checking the pid file contents.
running ps to look for nginx processes.

### task two, medium, hidden sparse log file filling a loopback mount

base filesystem contents.
a loopback mounted filesystem at a specific mount point with very limited space.
a hidden sparse log file consuming all available space on the mount.
the log file is in a nonobvious location with a misleading name.

grading function.
check if the agent identified the full filesystem, partial credit.
check if the agent found the hidden file, partial credit.
check if the agent freed space and the filesystem has available capacity, full credit.

diagnostic knowledge rewards.
running df to see disk usage.
running du to find large files.
running find to locate hidden files.
using lsof to find open file descriptors.

### task three, hard, broken network namespace with corrupted routing tables

base filesystem contents.
a network namespace with interfaces configured but routing tables corrupted.
the default route is missing or points to a nonexistent gateway.
dns resolution is broken.

grading function.
check if the agent diagnosed the routing issue, partial credit.
check if the default route is restored, partial credit.
check if dns resolution works, partial credit.
check if outbound connectivity is restored, full credit.

diagnostic knowledge rewards.
running ip route show to inspect routing.
running ip addr to check interface status.
running ip link to check link state.
running ping to test connectivity.
checking resolv.conf for dns configuration.

validation for all tasks.
deploy each task in a sandbox and confirm the fault condition exists.
manually remediate each task and confirm the grading function awards full credit.
reset each task and confirm it returns to the broken state.


---

## phase five, reward shaping engine

goal is to implement the potential based reward shaping system. depends on phase four for task grading functions.

deliverables.
implement the reward engine class in rewards.py.
maintain a knowledge state set per episode tracking discovered diagnostic facts.
define the diagnostic fact triggers for each task as a mapping from command patterns to fact identifiers.
compute knowledge delta as a fractional reward for each newly discovered fact.
compute health delta by calling the task grading function and comparing to the previous health score.
apply a constant negative step penalty to discourage unnecessary commands.
detect catastrophic actions using a blocklist of destructive command patterns including rm rf on critical paths, killing init or pid 1, overwriting boot or etc with garbage.
return a large negative reward and episode termination for catastrophic actions.

validation.
simulate a sequence of diagnostic commands for each task and verify monotonically increasing knowledge rewards.
simulate a remediation sequence and verify health delta tracks progress.
trigger a catastrophic action and verify the large negative penalty and episode termination.


---

## phase six, fastapi server and websocket handler

goal is to build the central server that ties all components together. depends on all previous phases.

deliverables.
implement the fastapi application in server.py.
a websocket endpoint at /ws that accepts agent connections.
on connection, initialize a new episode by selecting a task and creating a sandbox.
receive action json from the agent, validate against the action pydantic model.
execute the command in the sandbox.
construct the observation using the sandbox output and the reward engine.
send the observation json back to the agent.
on episode completion, log results, reset the sandbox, and optionally start a new episode.
add a health check endpoint at /health for monitoring.
add a task list endpoint at /tasks returning available scenarios.

the server must handle.
agent disconnection gracefully, cleaning up the sandbox.
command execution timeouts without crashing.
malformed action json with appropriate error responses.

validation.
start the server and connect with a websocket client.
send a valid action and receive a valid observation.
send an invalid action and receive an error response.
disconnect and confirm the sandbox is cleaned up.
run a full episode from start to finish.


---

## phase seven, inference script and openenv compliance

goal is to build the inference.py entry point that connects to the server, runs the agent loop, and produces correctly tagged output. depends on phase six.

deliverables.
implement inference.py to read environment variables for api key and model name.
connect to the fastapi websocket endpoint.
implement the agent loop that sends observations to the language model api and relays the model actions to the websocket.
wrap all output in openenv bracket tags.
start tag format is, bracket open, start, bracket close.
step tag format is, bracket open, step, bracket close, followed by the action and observation.
end tag format is, bracket open, end, bracket close.
handle api errors, timeouts, and rate limits gracefully.
implement a fallback heuristic agent for cases where the api is unavailable.

validation.
run inference.py against the running server.
verify the output contains correct openenv bracket tags.
verify the agent completes at least one episode.
verify the total runtime stays within the 20 minute budget.


---

## phase eight, integration testing and hardening

goal is to run the complete system end to end and fix any issues. depends on all previous phases.

deliverables.
run a full evaluation across all three tasks sequentially.
measure total wall clock time and confirm it is under 20 minutes.
measure peak memory usage and confirm it is under 8 gb.
measure cpu utilization and confirm no sustained spikes above 2 vcpu equivalent.
fix any race conditions, resource leaks, or error handling gaps.
add defensive timeout wrappers around all external calls.
test graceful degradation when the language model api is slow or unavailable.

validation.
three consecutive clean runs with no crashes.
all timing and resource constraints met.
correct openenv bracket tag output for all episodes.


---

## phase nine, deployment and submission

goal is to package the system for hugging face spaces and submit.

deliverables.
create or verify the hugging face space configuration.
ensure all dependencies are in requirements.txt with pinned versions.
verify the openenv.yaml manifest is complete and correct.
test the deployment in a fresh environment.
submit the solution.

validation.
the space builds and starts without errors.
the inference script runs and produces valid output.
the total execution completes within the 20 minute window.

