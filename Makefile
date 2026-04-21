PYTHON ?= python
MODEL ?= google/gemma-4-e4b-it
GROUP_SIZE ?= 4
MAX_TURNS ?= 12
NUM_STEPS ?= 100
SCENARIOS ?= hpc_outage,hpc_munge,hpc_pid_stale
ENV_URLS ?=
RUN_DIR ?= ./runs/hpc_grpo

.PHONY: help install bench gold eval demo train train-remote dry dry-remote serve clean

help:
	@echo "Targets for EnterpriseHPC-v0"
	@echo "  make install       install python dependencies"
	@echo "  make bench         reset-latency benchmark (200 iterations)"
	@echo "  make gold          prove every scenario is solvable (deterministic)"
	@echo "  make eval          run gold/random/bad policies + leaderboard.md"
	@echo "  make demo          gold trajectory run with transcripts printed"
	@echo "  make dry           local dry-run training rollout (no gpu)"
	@echo "  make dry-remote    dry-run against a hosted openenv space (set ENV_URLS=...)"
	@echo "  make train         full grpo training locally with gemma-4-e4b-it"
	@echo "  make train-remote  full grpo training against ENV_URLS (hf spaces)"
	@echo "  make serve         run the openenv server on :8000"
	@echo "  make clean         remove runs/ caches"

install:
	$(PYTHON) -m pip install -r requirements.txt

bench:
	$(PYTHON) -m bench.bench_reset -n 200

gold:
	$(PYTHON) -m tools.verify_gold_trajectory -v

eval:
	$(PYTHON) -m eval.eval_suite --trials 3 --scenarios $(SCENARIOS) --output-dir ./runs/eval

demo: gold
	@echo "see docs/pitch.md for the 3-minute demo script"

dry:
	$(PYTHON) -m training.train_hpc_outage --dry-run \
	  --group-size $(GROUP_SIZE) --max-turns $(MAX_TURNS) \
	  --scenarios $(SCENARIOS) --output-dir $(RUN_DIR)

dry-remote:
	@test -n "$(ENV_URLS)" || (echo "set ENV_URLS=https://... to target a hosted space" && exit 1)
	$(PYTHON) -m training.hpc_openenv_gemma --dry-run \
	  --env-urls $(ENV_URLS) --group-size $(GROUP_SIZE) --max-turns $(MAX_TURNS) \
	  --scenarios $(SCENARIOS) --output-dir $(RUN_DIR)

train:
	$(PYTHON) -m training.train_hpc_outage \
	  --model $(MODEL) --group-size $(GROUP_SIZE) --max-turns $(MAX_TURNS) \
	  --num-train-steps $(NUM_STEPS) --scenarios $(SCENARIOS) \
	  --output-dir $(RUN_DIR)

train-remote:
	@test -n "$(ENV_URLS)" || (echo "set ENV_URLS=https://... to target a hosted space" && exit 1)
	$(PYTHON) -m training.hpc_openenv_gemma \
	  --env-urls $(ENV_URLS) --model $(MODEL) \
	  --group-size $(GROUP_SIZE) --max-turns $(MAX_TURNS) \
	  --num-train-steps $(NUM_STEPS) --scenarios $(SCENARIOS) \
	  --output-dir $(RUN_DIR)

serve:
	uv run server --host 0.0.0.0 --port 8000

clean:
	rm -rf runs __pycache__ **/__pycache__ .pytest_cache
