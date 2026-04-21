from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--env-urls",
        nargs="+",
        required=True,
        help="one or more openenv sysadmin server base urls. hosted hf spaces work directly",
    )
    parser.add_argument("--model", default=os.environ.get("HPC_MODEL", "google/gemma-4-e4b-it"))
    parser.add_argument("--output-dir", default="./runs/hpc_openenv_gemma")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--num-train-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenarios", default="hpc_outage,hpc_munge,hpc_pid_stale")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--hub-repo", default=os.environ.get("HF_HUB_REPO"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="skip heavy deps and run a single random-policy rollout through the remote servers",
    )
    parser.add_argument(
        "--backend",
        choices=["unsloth", "transformers"],
        default="unsloth",
        help="model loader. unsloth (default) for colab/single gpu, transformers for vertex/hf jobs",
    )
    return parser.parse_args()


def _resolve_scenarios(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("at least one scenario id must be provided")
    return names


def _random_policy(rng: random.Random):
    pool = [
        "sinfo",
        "squeue",
        "ssh compute-01",
        "cat /etc/sysconfig/network-scripts/route-eth0",
        "printf 'default via 10.0.0.1 dev eth0\\n10.0.0.0/24 dev eth0 proto kernel scope link src 10.0.0.11\\n' > /etc/sysconfig/network-scripts/route-eth0",
        "systemctl restart slurmd",
        "chmod 0400 /etc/munge/munge.key",
        "systemctl restart munge",
        "exit",
        "curl -I http://localhost:8080/",
    ]

    def generate(batches):
        return [f"<bash>{rng.choice(pool)}</bash>" for _ in batches]

    return generate


def _env_factory(env_urls: list[str], scenarios: list[str]):
    from training.remote_env import HttpEnterpriseHPCEnv
    from training.remote_env import RemoteEndpointPool

    pool = RemoteEndpointPool(env_urls)

    def make_env():
        return HttpEnterpriseHPCEnv(env_urls=env_urls, scenario_pool=scenarios, pool=pool)

    return make_env, pool


def _dry_run(args: argparse.Namespace) -> int:
    from training.logger import RewardLogger
    from training.rollout import run_interactive_group
    from training.rollout import summarize_group

    scenarios = _resolve_scenarios(args.scenarios)
    rng = random.Random(args.seed)
    make_env, pool = _env_factory(args.env_urls, scenarios)
    logger = RewardLogger(args.output_dir, run_name="dry_run", hub_repo=args.hub_repo, wandb_project=args.wandb_project)

    try:
        records = run_interactive_group(
            group_size=args.group_size,
            generate_fn=_random_policy(rng),
            env_factory=make_env,
            max_turns=args.max_turns,
            seed_start=args.seed,
        )
        logger.log(step=0, records=records)
        print(f"dry_run summary {summarize_group(records)}")
    finally:
        logger.close()
        pool.close()
    return 0


def _load_model_and_tokenizer(args: argparse.Namespace):
    if args.backend == "unsloth":
        try:
            from unsloth import FastLanguageModel  # type: ignore

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model,
                max_seq_length=args.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=args.lora_alpha,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )
            FastLanguageModel.for_inference(model)
            return model, tokenizer, "unsloth"
        except ImportError:
            print("unsloth missing falling back to transformers backend", file=sys.stderr)

    import torch  # type: ignore
    from peft import LoraConfig  # type: ignore
    from peft import get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    try:
        from transformers import AutoModelForMultimodalLM  # type: ignore

        model = AutoModelForMultimodalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    return model, tokenizer, "transformers"


def _train(args: argparse.Namespace) -> int:
    try:
        from datasets import Dataset  # type: ignore
        from trl import GRPOConfig  # type: ignore
        from trl import GRPOTrainer  # type: ignore
    except ImportError as exc:
        print(f"trl or datasets missing install them first {exc}", file=sys.stderr)
        return 2

    import torch  # type: ignore

    from training.agent_prompt import SYSTEM_PROMPT
    from training.agent_prompt import USER_PROMPT
    from training.logger import RewardLogger
    from training.rollout import run_interactive_group
    from training.rollout import summarize_group

    scenarios = _resolve_scenarios(args.scenarios)
    make_env, pool = _env_factory(args.env_urls, scenarios)

    print(f"train load model {args.model} backend {args.backend}")
    model, tokenizer, backend = _load_model_and_tokenizer(args)

    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    dataset = Dataset.from_dict({"prompt": [prompt_text] * max(args.num_train_steps, 32)})

    def generate_fn(batch_messages):
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    logger = RewardLogger(
        args.output_dir,
        run_name="hpc_openenv_gemma",
        hub_repo=args.hub_repo,
        wandb_project=args.wandb_project,
    )
    step_counter = {"n": 0}

    def compute_environment_reward(prompts, completions, **kwargs):
        step_counter["n"] += 1
        records = run_interactive_group(
            group_size=len(completions),
            generate_fn=generate_fn,
            env_factory=make_env,
            max_turns=args.max_turns,
            seed_start=random.randrange(1 << 30),
        )
        summary = summarize_group(records)
        logger.log(step=step_counter["n"], records=records)
        print(f"grpo group summary {summary}")
        return [float(r.reward) for r in records]

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=args.group_size,
        max_prompt_length=args.max_seq_length // 2,
        max_completion_length=args.max_new_tokens,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.num_train_steps,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=(not torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        report_to=args.report_to,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[compute_environment_reward],
        args=training_args,
        train_dataset=dataset,
    )

    try:
        print(f"train start backend {backend} steps {args.num_train_steps} group {args.group_size}")
        started = time.time()
        trainer.train()
        print(f"train done elapsed {time.time() - started:.1f}s")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    finally:
        logger.close()
        pool.close()
    return 0


def main() -> int:
    args = _parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        return _dry_run(args)
    return _train(args)


if __name__ == "__main__":
    raise SystemExit(main())
