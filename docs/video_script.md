# 2 minute video script: EnterpriseHPC-v0

target length 110 seconds. shots labeled A through F. copy the voice
over into a teleprompter, screen record with asciinema while narrating.

## shot A, 0:00–0:10, title card

> "can a language model run an hpc cluster? we built EnterpriseHPC-v0
> to find out."

screen: repo readme header with the architecture diagram.

## shot B, 0:10–0:30, the incident

> "open ondemand returns five oh two. the compute partition is
> drained. a cfd job is stuck in pending auth fail. this is a real
> enterprise sre incident and we reproduce every signal of it inside
> a single unprivileged sandbox."

screen: split terminal showing `sinfo` drain, `squeue` pending,
`curl -I http://localhost:8080` returning 502 Bad Gateway.

## shot C, 0:30–0:55, architecture in one sentence

> "no docker, no virtual machines. just bubblewrap with fuse
> overlayfs on tmpfs for two millisecond resets, nested bwrap for
> ssh lateral movement, and a mock slurm state machine that the
> stubbed binaries read under fcntl locks."

screen: left pane `python -m bench.bench_reset -n 100`, highlight
p50 2.40 ms. right pane `tree nodes/` showing login and compute-01.

## shot D, 0:55–1:25, the agent loop

> "google gemma four e four b it, trained with trl grpo on a single
> gpu. the reward is binary. the grader reads explicit filesystem
> state. no reward hacking. watch the trained agent take the
> remediation path end to end."

screen: speed ramp the following commands, one per prompt switch:
`sinfo`, `ssh compute-01`, `cat route-eth0`, `printf default via
10.0.0.1 ... > route-eth0`, `systemctl restart slurmd`, `exit`,
`curl -I http://localhost:8080` flipping to 200 OK.

## shot E, 1:25–1:45, reward curve

> "solve rate climbs from zero to seventy percent across a hundred
> grpo steps on three scenarios, hpc outage, hpc munge, and hpc
> pid stale. the agent does not just memorize, it routes between
> fault modes."

screen: tensorboard reward curve from `runs/hpc_grpo` with
solve_rate overlaid.

## shot F, 1:45–1:55, call to action

> "spec, code, blog, space, colab. links in the description. go
> break something and teach a model to fix it."

screen: endcard with repo url, hf space url, colab url, blog url.
