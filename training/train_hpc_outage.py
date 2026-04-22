from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default=os.environ.get("HPC_MODEL", "google/gemma-4-e4b-it"),
        help="hf hub id. defaults to google/gemma-4-e4b-it. swap to google/gemma-4-e2b-it for t4 colab",
    )
    parser.add_argument("--output-dir", default="./runs/hpc_grpo")
    parser.add_argument("--group-size", type=int, default=4, help="grpo num_generations")
    parser.add_argument("--max-turns", type=int, default=16, help="max interaction turns per rollout")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--num-train-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=384, help="per turn generation cap")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--scenarios",
        default="hpc_outage,hpc_munge,hpc_pid_stale,hpc_gpu_ecc,hpc_nfs_stale,hpc_ood_apache",
        help="comma separated task ids to sample from",
    )
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="skip heavy imports and run one rollout with a random policy")
    parser.add_argument("--report-to", default="tensorboard")
    return parser.parse_args()


def _resolve_scenarios(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("at least one scenario id must be provided")
    return names


def _random_policy_generator(rng: random.Random):
    pool = [
        "sinfo",
        "squeue",
        "ssh compute-01",
        "cat /etc/sysconfig/network-scripts/route-eth0",
        "printf 'default via 10.0.0.1\\n' > /etc/sysconfig/network-scripts/route-eth0",
        "systemctl restart slurmd",
        "chmod 0400 /etc/munge/munge.key",
        "systemctl restart munge",
        "rm /var/run/slurmd.pid",
        "nvidia-smi",
        "nvidia-smi -r -i 0",
        "umount -l /mnt/shared",
        "mount /mnt/shared",
        "apachectl configtest",
        "apachectl graceful",
        "exit",
        "curl -I http://localhost:8080/",
        "curl -I http://localhost:8081/",
    ]

    def generate(batches):
        return [f"<bash>{rng.choice(pool)}</bash>" for _ in batches]

    return generate


def _dry_run(args: argparse.Namespace) -> int:
    from hpc_gym import EnterpriseHPCEnv
    from training.rollout import run_interactive_group
    from training.rollout import summarize_group

    rng = random.Random(args.seed)
    scenarios = _resolve_scenarios(args.scenarios)
    generate_fn = _random_policy_generator(rng)

    def env_factory() -> EnterpriseHPCEnv:
        return EnterpriseHPCEnv(scenario_pool=scenarios)

    records = run_interactive_group(
        group_size=args.group_size,
        generate_fn=generate_fn,
        env_factory=env_factory,
        max_turns=args.max_turns,
        seed_start=args.seed,
    )
    summary = summarize_group(records)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "dry_run_summary.json").write_text(
        json.dumps({"summary": summary, "records": [r.__dict__ for r in records]}, default=str, indent=2)
    )
    print(f"dry_run summary {summary}")
    return 0


def _train(args: argparse.Namespace) -> int:
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError as exc:
        print(f"unsloth missing install unsloth first {exc}", file=sys.stderr)
        return 2

    try:
        from datasets import Dataset  # type: ignore
        from trl import GRPOConfig  # type: ignore
        from trl import GRPOTrainer  # type: ignore
    except ImportError as exc:
        print(f"trl or datasets missing install them first {exc}", file=sys.stderr)
        return 2

    import torch  # type: ignore

    from hpc_gym import EnterpriseHPCEnv
    from training.rollout import run_interactive_group
    from training.rollout import summarize_group

    scenarios = _resolve_scenarios(args.scenarios)

    print(f"train load model {args.model} max_seq {args.max_seq_length}")
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

    from training.agent_prompt import SYSTEM_PROMPT
    from training.agent_prompt import USER_PROMPT

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
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in batch_messages
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def env_factory() -> EnterpriseHPCEnv:
        return EnterpriseHPCEnv(scenario_pool=scenarios)

    rollout_buffer: dict[str, Any] = {"records": []}

    def compute_environment_reward(prompts, completions, **kwargs):
        records = run_interactive_group(
            group_size=len(completions),
            generate_fn=generate_fn,
            env_factory=env_factory,
            max_turns=args.max_turns,
            seed_start=random.randrange(1 << 30),
        )
        rollout_buffer["records"] = records
        summary = summarize_group(records)
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

    print(f"train start steps {args.num_train_steps} group {args.group_size}")
    started = time.time()
    trainer.train()
    elapsed = time.time() - started
    print(f"train done elapsed {elapsed:.1f}s")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


def main() -> int:
    args = _parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        return _dry_run(args)
    return _train(args)


if __name__ == "__main__":
    raise SystemExit(main())
