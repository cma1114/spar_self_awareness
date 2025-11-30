import argparse
import math
import numpy as np
import os
import torch
import wandb
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

# Imports from helper files
from finetune_evaluation_metrics import (
    # compute_metrics_for_wandb,
    #  assess_mcq_accuracy,
      run_evaluation,
    #    log_answer_distributions
)
from finetune_utils import (
    # write_log,
    # get_single_token_id,
    # init_wandb,
    # log_wandb_metrics,
    # log_wandb_config,
    # log_device_info,
    # save_hf_checkpoint,
    # save_model_final,
    # finish_wandb,
    build_self_confidence_prompts,
    build_multiple_choice_question_prompts,
    # _get_log_file_path,
    # check_and_clear_gpu_memory,
    # load_model_with_error_handling,
    # setup_tokenizer,
    # log_prompts_and_responses,
    # compute_ABCD_entropy,
    # shuffle_options_and_update_correct_letter,
    # parse_letter_from_model_text,
    run_mcq_forward_pass,
    run_confidence_forward_pass,
    get_single_token_id,
    load_tokenizer,
    load_model_with_lora,
    convert_entropy_to_soft_labels,
    prepare_model_and_tokenizer
)
from finetune_data_handling import (
    # MCQDataset,
    # load_mcq_results_data,
    # verify_and_resolve_options,
    # validate_and_load_dataset,
    # validate_training_files,
    # write_jsonl,
    get_batch,
    load_jsonl_dataset,
    collate_fn
)



# ============================================================
# Entropy → scalar confidence → soft labels
# ============================================================

def compute_soft_labels(logits4, sigma=10.0):
    """
    Convert 4-way answer logits into soft 8-bin confidence distribution.

    Uses percentage-based Gaussian kernel to create soft labels.

    Args:
        logits4: tensor of shape [4] with logits for A, B, C, D
        sigma: Gaussian width in percentage space (default: 10)

    Returns:
        tensor of shape [8] with soft label distribution
    """
    # 1. Softmax over the 4 MCQ options
    probs = torch.softmax(logits4, dim=0)

    # 2. Entropy (natural logs)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()

    # 3. Convert entropy to "confidence percentage"
    #    confidence = (1 - H/log(4)) * 100
    confidence_percent = (1 - entropy / math.log(4)) * 100.0

    # 4. Bin midpoints + widths (exact values from your colleague)
    bin_edges = torch.tensor([0, 5, 10, 20, 40, 60, 80, 90, 100],
                             dtype=torch.float32,
                             device=logits4.device)
    bin_midpoints = torch.tensor([2.5, 7.5, 15, 30, 50, 70, 85, 95],
                                 dtype=torch.float32,
                                 device=logits4.device)
    bin_widths = bin_edges[1:] - bin_edges[:-1]   # shape [8]

    # 5. Gaussian kernel in percentage space
    distances = (bin_midpoints - confidence_percent)**2
    weights = torch.exp(-distances / (2 * sigma * sigma)) * bin_widths

    return weights / weights.sum()


# ------------------------------------------------------------------
# Training Step
# ------------------------------------------------------------------
def train_step(model, tokenizer, batch, device, sigma, args):
    model.train()

    # ----------------------------------------------
    # 1. MCQ forward pass (TRAIN VERSION)
    # ----------------------------------------------
    mcq_prompts = build_multiple_choice_question_prompts(batch, tokenizer)
    enc = tokenizer(mcq_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    out = model(**enc)
    final_logits = out.logits[:, -1, :]

    # Extract A–D logits
    abcd_ids = torch.tensor(
        [get_single_token_id(tokenizer, c) for c in "ABCD"],
        device=device
    )
    logits4 = final_logits[:, abcd_ids]       # [B,4]

    # Compute entropy for soft labels (NO .item())
    probs4 = torch.softmax(logits4, dim=-1)
    entropy = -(probs4 * torch.log(probs4 + 1e-12)).sum(dim=-1)   # [B]

    # Soft labels (tensor)
    soft = convert_entropy_to_soft_labels(entropy)                # [B,8]

    # ----------------------------------------------
    # 2. Confidence forward pass (TRAIN VERSION)
    # ----------------------------------------------
    conf_prompts = build_self_confidence_prompts(batch, tokenizer)
    enc2 = tokenizer(conf_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    out2 = model(**enc2)

    final_logits2 = out2.logits[:, -1, :]
    bins_ids = torch.tensor(
        [get_single_token_id(tokenizer, c) for c in "ABCDEFGH"],
        device=device
    )
    logits8 = final_logits2[:, bins_ids]      # [B,8]

    # ----------------------------------------------
    # 3. Compute loss (NO item())
    # ----------------------------------------------
    log_p = torch.log_softmax(logits8, dim=-1)
    loss = -(soft * log_p).sum(dim=-1).mean()

    # ----------------------------------------------
    # 4. Backprop
    # ----------------------------------------------
    loss.backward()

    return loss.detach()


# ============================================================
# Main training
# ============================================================


def train(args):
    """
    Trainer for Expected Confidence Task (ECT).
    Uses:
        - run_mcq_forward_pass()
        - run_confidence_forward_pass()
        - train_step()
        - val_step()
        - run_evaluation()
    """

    # ============================================================
    # Setup / Load model
    # ============================================================
    device = args.device
    tokenizer = load_tokenizer(args)
    model = load_model_with_lora(args, tokenizer).to(device)

    # Canonicalize model/tokenizer setup (fix pad_token warnings)
    model, tokenizer = prepare_model_and_tokenizer(model, tokenizer)


    # Dataset loading ------------------------------------------------
    train_dataset = load_jsonl_dataset(args.train_data_path)
    val_dataset   = load_jsonl_dataset(args.val_data_path)

    print(f"✓ Training dataset loaded: {len(train_dataset)} samples")
    print(f"✓ Validation dataset loaded: {len(val_dataset)} samples")
    print(f"  Validation will run every {args.val_interval} steps")

    # Optimizer ------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0
    )

    # Logging --------------------------------------------------------
    if args.save_wandb_artifact:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

    # Output / checkpoints
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Setup evaluation log file path
    log_dir = "finetune_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    model_name_safe = args.model_name.replace("/", "-").replace("_", "-")
    log_file_path = os.path.join(log_dir, f"{timestamp}_{model_name_safe}_evaluation_metrics.jsonl")

    # ============================================================
    # Baseline evaluation BEFORE training
    # ============================================================
    print("\n" + "="*60)
    print("Running baseline validation (before training)...")
    print("="*60)

    baseline_metrics = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        device=device,
        args=args,
        step_name="baseline",
        num_samples=args.val_num_samples,
        log_file_path=log_file_path,
        step=0,
    )

    print(f"\nBaseline Accuracy: {baseline_metrics['mcq_accuracy']:.4f}")
    print(f"Baseline Avg Entropy: {baseline_metrics['avg_entropy']:.4f}")
    print(f"Baseline Avg Confidence: {baseline_metrics['avg_confidence']:.4f}")
    print(f"- samples: {baseline_metrics['n_samples']}\n")

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    step = 0
    losses = []

    while step < args.max_steps:
        batch = get_batch(train_dataset, args.batch_size)

        # -----------------------------
        # Train step 
        # -----------------------------
        optimizer.zero_grad()
        loss = train_step(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            sigma=args.sigma,
            device=device,
            args=args,
        )
        optimizer.step()

        losses.append(loss.item())

        # W&B logging
        if args.save_wandb_artifact:
            wandb.log({"train/loss": loss.item(), "step": step})

        # -----------------------------
        # Periodic validation (Validation Step)
        # -----------------------------
        if (step % args.val_interval) == 0 and step > 0:
            print("\n" + "="*60)
            print(f"Validation at step {step}")
            print("="*60)

            val_metrics = run_evaluation(
                model=model,
                tokenizer=tokenizer,
                val_dataset=val_dataset,
                device=device,
                args=args,
                step_name="validation",
                num_samples=args.val_num_samples,
                log_file_path=log_file_path,
                step=step,
            )

            print(f"Val Accuracy: {val_metrics['mcq_accuracy']:.4f}")
            print(f"Val Loss:     {val_metrics['avg_loss']:.4f}")
            print(f"Val Entropy:  {val_metrics['avg_entropy']:.4f}")
            print(f"Val Conf:     {val_metrics['avg_confidence']:.4f}")
            print(f"Samples:      {val_metrics['n_samples']}")

        # -----------------------------
        # Periodic checkpointing
        # -----------------------------
        if (step % args.checkpoint_steps) == 0 and step > 0:
            ckpt_dir = os.path.join(output_dir, f"ckpt_step_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f"Saving checkpoint at step {step} → {ckpt_dir}")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

        step += 1

    # ============================================================
    # Final metrics
    # ============================================================
    print("\n" + "="*60)
    print("Final evaluation:")
    print("="*60)

    final_metrics = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        device=device,
        args=args,
        step_name="final",
        num_samples=args.val_num_samples,
        log_file_path=log_file_path,
        step=step,
    )

    print(f"\nFinal Accuracy:  {final_metrics['mcq_accuracy']:.4f}")
    print(f"Final Loss:      {final_metrics['avg_loss']:.4f}")
    print(f"Final Confidence:{final_metrics['avg_confidence']:.4f}")

    if args.save_wandb_artifact:
        wandb.finish()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train dynamic metacognition model "
                    "(Explicit Confidence Task)"
    )

    # -----------------------
    # Model
    # -----------------------
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HF model name or path")
    parser.add_argument("--device", type=str,
                        default="cuda",
                        choices=["cuda", "cpu"],
                        help="Compute device")

    # -----------------------
    # Data
    # -----------------------
    parser.add_argument("--train_data_path", type=str,
                        required=True,
                        help="Path to JSONL training dataset")
    
    parser.add_argument("--val_data_path", type=str,
                        default=None,
                        help="Path to JSONL validation dataset (optional)")
    
    parser.add_argument("--test_data_path", type=str,
                        default=None,
                        help="Path to JSONL test dataset (optional, will be evaluated after training)")

    parser.add_argument("--batch_size", type=int,
                        default=4,
                        help="Training batch size")

    parser.add_argument("--mcq_results_data", type=str,
                        default=None,
                        help="Path to JSON/JSONL file with previous MCQ results for verification")

    # -----------------------
    # LoRA
    # -----------------------
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="Modules to apply LoRA to")

    # -----------------------
    # Training
    # -----------------------
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=1000000000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=100,
                        help="Run validation every N training steps")
    parser.add_argument("--limit_val_batches", type=int, default=None,
                        help="Limit validation to N batches (None = use all validation data)")
    parser.add_argument("--val_num_samples", type=int, default=500,
                        help="Number of random questions to sample from validation dataset for validation steps (default: 500)")
    parser.add_argument("--sigma", type=float, default=10.0,
                        help="Sigma parameter for soft label distribution")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for sampling predictions (0.0 = deterministic/argmax, >0 = sampling)")
    parser.add_argument(
        "--no_shuffle_options", dest="shuffle_options", action="store_false",
        help="Disable shuffling of multiple choice answer options (shuffling is enabled by default)"
    )
    parser.add_argument(
        "--use_recorded_responses", action="store_true", default=None,
        help=("Use recorded MCQ responses (frozen teacher) as training "
              "targets instead of recomputing logits.")
    )
    parser.add_argument(
        "--no_use_recorded_responses", dest="use_recorded_responses",
        action="store_false",
        help=("Disable using recorded responses, use dynamic teacher "
              "(current model logits) instead.")
    )


    # -----------------------
    # Output
    # -----------------------
    parser.add_argument("--output_dir", type=str,
                        default="dynamic_ect_lora",
                        help="Directory to save final model")

    parser.add_argument("--save_hf", action="store_true",
                        help="If set, push LoRA model to HuggingFace Hub")

    parser.add_argument("--hf_repo", type=str,
                        default=None,
                        help="HF repo name if pushing to Hub")

    parser.add_argument("--save_hf_checkpoints", action="store_true",
                        help="If set, save checkpoints to HuggingFace Hub during training")

    parser.add_argument("--hf_checkpoint_repo", type=str,
                        default=None,
                        help="Base HF repo name for checkpoints (e.g., 'username/model-name')")

    parser.add_argument("--checkpoint_steps", type=int,
                        default=500,
                        help="Save checkpoint every N steps")

    parser.add_argument("--hf_checkpoint_private", action="store_true",
                        help="If set, make checkpoint repos private")

    # -----------------------
    # Weights & Biases
    # -----------------------
    parser.add_argument("--wandb_project", type=str,
                        default="llm-metacognition-ect",
                        help="W&B project name")
    
    parser.add_argument("--wandb_run_name", type=str,
                        default=None,
                        help="W&B run name (auto-generated if not provided)")
    
    parser.add_argument("--wandb_tags", type=str, nargs="+",
                        default=None,
                        help="Tags for W&B run")
    
    parser.add_argument("--wandb_notes", type=str,
                        default=None,
                        help="Notes/description for W&B run")
    
    parser.add_argument("--save_wandb_artifact", action="store_true",
                        help="Save model as W&B artifact for reproducibility")

    args = parser.parse_args()
    
    # Set default for shuffle_options (True if --no_shuffle_options was not provided)
    if not hasattr(args, 'shuffle_options'):
        args.shuffle_options = True
    
    # Validate that exactly one of --use_recorded_responses or --no_use_recorded_responses is set
    if args.use_recorded_responses is None:
        parser.error(
            "Exactly one of --use_recorded_responses or --no_use_recorded_responses must be specified. "
            "You must explicitly choose whether to use recorded responses (frozen teacher) or live responses (dynamic teacher)."
        )
    
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
