"""
DPO training with LoRA on Apple Silicon MPS.

Usage:
  python3 train_dpo.py --base-model models/Llama-2-7b-hf --dataset ../data/multidomain_preferences.json --output-dir ../results/dpo_multidomain
  python3 train_dpo.py --base-model models/Llama-3.1-8B --dataset ../data_llama31/multidomain_preferences.json --output-dir ../results_llama31/dpo_multidomain
"""
import os
import json
import argparse
import torch

os.environ["WANDB_DISABLED"] = "true"

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    with open(args.dataset) as f:
        ori_dataset = json.load(f)
    print(f"Preference pairs: {len(ori_dataset)}")

    if not ori_dataset:
        print("ERROR: No preference pairs found.")
        return

    data_dict = {key: [item[key] for item in ori_dataset] for key in ori_dataset[0]}
    dataset = Dataset.from_dict(data_dict)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, low_cpu_mem_usage=True, torch_dtype=torch.float32,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    effective_bs = args.bs * args.grad_accum
    steps_per_epoch = max(1, len(split["train"]) // effective_bs)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(10, int(0.1 * steps_per_epoch))
    save_interval = max(1, total_steps // 3)

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=save_interval,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        eval_strategy="steps",
        eval_steps=min(50, steps_per_epoch),
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_steps=warmup,
        optim="adamw_torch",
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        beta=args.beta,
        max_length=args.max_length,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out", "wte"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    dpo_trainer.train()

    final_dir = os.path.join(args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()
