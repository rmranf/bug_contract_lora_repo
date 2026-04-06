from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a contract-first LoRA adapter on the qwen3.5 executor-aligned dataset."
    )
    parser.add_argument("--model-path", required=True, help="Local HF model path or model id.")
    parser.add_argument(
        "--train-data",
        required=True,
        help="JSONL path produced by execution_contract_training_chat_v1 or mixed_goldx10.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save LoRA outputs.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable QLoRA-style 4bit base model loading via bitsandbytes.",
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        default="nf4",
        choices=["nf4", "fp4"],
        help="4bit quantization type when --load-in-4bit is enabled.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for lower memory use.",
    )
    parser.add_argument(
        "--target-modules",
        nargs="*",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        help="PEFT target modules for LoRA.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Force bf16 training. If omitted, auto-detect is used.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 training. If omitted, auto-detect is used.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_chat_text(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


class JsonlChatDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: AutoTokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        text = build_chat_text(self.tokenizer, row["messages"])
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = list(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class CausalLMCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids = []
        attention_mask = []
        labels = []
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def infer_torch_dtype(args: argparse.Namespace) -> torch.dtype | None:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return None


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise SystemExit(
            "peft is required for LoRA training. Install it first, then rerun this script."
        ) from exc

    train_rows = load_jsonl(Path(args.train_data))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = infer_torch_dtype(args)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "bitsandbytes is required for --load-in-4bit QLoRA. Install it first, then rerun."
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch_dtype or torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, peft_config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    train_dataset = JsonlChatDataset(train_rows, tokenizer, args.max_length)
    collator = CausalLMCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    run_manifest = {
        "model_path": args.model_path,
        "train_data": str(Path(args.train_data).as_posix()),
        "output_dir": str(output_dir.as_posix()),
        "row_count": len(train_rows),
        "max_length": args.max_length,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.target_modules,
        },
        "qlora": {
            "load_in_4bit": args.load_in_4bit,
            "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
