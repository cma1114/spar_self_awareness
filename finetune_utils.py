import json
import math
import os
import re
import random
import numpy as np
import traceback
from datetime import datetime, timezone
from torch.utils.data import Dataset
import torch
import string

from finetune_data_handling import (
    # write_jsonl,
    # get_batch,
    # load_jsonl_dataset,
    # MCQDataset,
    # normalize_text,
    # load_mcq_results_data,
    # verify_and_resolve_options,
    # validate_file_exists_and_not_empty,
    # validate_and_load_dataset,
    # resolve_file_path,
    # validate_training_files,
    collate_fn as data_collate_fn
)

def load_tokenizer(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        padding_side="left"
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def prepare_model_and_tokenizer(model, tokenizer):
    """
    Standardizes tokenizer + model config for LLaMA.
    Ensures:
      • pad_token exists (LLaMA doesn't have one)
      • pad_token_id = eos_token_id
      • generation_config is consistent
      • padding/truncation behavior stable
    """

    # 1. LLaMA has no pad token → use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Model must know the pad token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 3. (Optional, recommended) unify generation config
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

    # 4. Normalize tokenizer behavior
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    return model, tokenizer



def load_model_with_lora(args, tokenizer):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    # LoRA
    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lcfg)

    return model

def run_mcq_forward_pass(model, tokenizer, prompts, device="cuda", temperature=0.0):
    """
    MCQ pass:
      • One forward pass
      • Extract A–D logits
      • Argmax for behavioral answer
      • Compute entropy
    """

    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = model(**enc, use_cache=False)

    final_logits = out.logits[:, -1, :]  # [B, vocab]

    abcd_ids = torch.tensor(
        [get_single_token_id(tokenizer, c) for c in "ABCD"],
        device=device,
        dtype=torch.long,
    )

    logits4 = final_logits[:, abcd_ids]   # [B, 4]
    probs4 = torch.softmax(logits4, dim=-1)
    entropy = -(probs4 * torch.log(probs4 + 1e-12)).sum(dim=-1)

    # Argmax prediction
    idx = logits4.argmax(dim=-1).tolist()
    pred_letters = [ "ABCD"[i] for i in idx ]

    # # In run_mcq_forward_pass()
    # print(f"DEBUG: Token IDs being used:")
    # for c in "ABCD":
    #     tid = get_single_token_id(tokenizer, c)
    #     decoded = tokenizer.decode([tid])
    #     print(f"  {c}: token_id={tid}, decodes_to='{decoded}'")

    # # In run_mcq_forward_pass(), after encoding:
    # print(f"DEBUG: Encoded prompt length: {enc['input_ids'].shape}")
    # print(f"DEBUG: Last 10 tokens: {enc['input_ids'][0, -10:].tolist()}")
    # print(f"DEBUG: Decoded last 10 tokens: {tokenizer.decode(enc['input_ids'][0, -10:])}")

    return {
        "pred_letters": pred_letters,
        "logits4": logits4,
        "probs4": probs4,
        "entropy": entropy,
    }



def run_confidence_forward_pass(
    model,
    tokenizer,
    prompts,
    device="cuda",
    temperature=0.0,
):
    """
    Confidence pass:
      • Extract A–H logits for bins → compute expected confidence
      • Generate 1 token → behavioral confidence prediction (A–H)
    """

    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    # ---- PASS 2: Logits ----
    with torch.no_grad():
        out = model(**enc, use_cache=False)
        final_logits = out.logits[:, -1, :]

    bins_ids = torch.tensor(
        [get_single_token_id(tokenizer, c) for c in "ABCDEFGH"],
        device=device
    )

    logits8 = final_logits[:, bins_ids]
    probs8 = torch.softmax(logits8, dim=-1)

    # Expected confidence (midpoints of bins)
    mids = torch.tensor([2.5, 7.5, 15, 30, 50, 70, 85, 95], dtype=torch.float32, device=device)
    expected_conf = (probs8 * mids).sum(dim=-1)

    # ---- PASS 2: Behavioral bin prediction ----
    gen = model.generate(
        **enc,
        max_new_tokens=1,
        do_sample=False,
        temperature=temperature,
        top_p=1.0
    )

    decoded = tokenizer.batch_decode(
        gen[:, enc["input_ids"].shape[1]:]
    )

    pred_bins = []
    for t in decoded:
        c = t.strip()[:1].upper()
        if c not in "ABCDEFGH":
            c = "?"
        pred_bins.append(c)

    return {
        "logits8": logits8,
        "probs8": probs8,
        "expected_conf": expected_conf,
        "pred_bins": pred_bins,
    }



# def parse_letter_from_model_text(model_text, valid_letters):
#     """
#     EXACTLY matches original logic in base_game_class.py:_parse_subject_decision.
#     """
#     if model_text is None:
#         return None

#     # Normalize
#     cleaned = model_text.upper().rstrip(string.whitespace + string.punctuation)

#     if len(cleaned) == 0:
#         return None

#     # Check first character
#     if cleaned[0] in valid_letters:
#         return cleaned[0]

#     # Check last character
#     if cleaned[-1] in valid_letters:
#         return cleaned[-1]

#     return None



def write_log(log_file_path, entry_dict):
    """
    Simple one-line logging function.

    Logs dict as JSON to file if path provided.

    Args:
        log_file_path: Path to log file (None to skip logging)
        entry_dict: Dictionary to log as JSON
    """
    if log_file_path:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry_dict, ensure_ascii=False) + '\n')


def _get_log_file_path(log_dir, model_name, suffix):
    """Helper function to create log file paths."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    model_name_safe = model_name.replace("/", "-").replace("_", "-")
    return os.path.join(
        log_dir, f"{model_name_safe}_{timestamp}_{suffix}.jsonl"
    )



def build_multiple_choice_question_prompts(batch, tokenizer):
    """Build MCQ prompts with proper chat template."""
    
    setup_prompt = (
        "I'm going to ask you a series of multiple-choice questions. "
        "For each one, select the answer you think is best. "
        "Respond only with the letter of your choice; do NOT output any other text."
    )
    
    prompts = []
    for row in batch:
        question = row["question"].strip()
        options = row.get("options", {})
        
        q_text = f"Question: {question}\n----------\n"
        for letter in "ABCD":
            q_text += f"{letter}: {options.get(letter, '')}\n"
        q_text += "------------------------------\nYour choice (A, B, C, or D):"
        
        # Apply chat template
        messages = [{"role": "user", "content": setup_prompt + "\n\n" + q_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    
    return prompts


def build_self_confidence_prompts(batch, tokenizer):
    """
    Build self-confidence prompts with proper chat template.
    The next token after the prompt should be one of A–H.
    """

    setup_prompt = (
        "I'm going to show you a multiple-choice question, and I want you to tell me "
        "your level of confidence that you would get the question right. Respond only "
        "with the letter corresponding to the percentage range you choose; do NOT output "
        "any other text."
    )

    # Confidence ranges A–H
    confidence_bins = {
        "A": "<5%",
        "B": "5-10%",
        "C": "10-20%",
        "D": "20-40%",
        "E": "40-60%",
        "F": "60-80%",
        "G": "80-90%",
        "H": ">90%",
    }

    nested_question_prompt = (
        "How confident are you that you would get this question right?"
    )

    prompts = []
    for row in batch:
        question = row["question"].strip()
        options = row.get("options", {})

        # Extract original MCQ options
        a_text = options.get("A", "").strip()
        b_text = options.get("B", "").strip()
        c_text = options.get("C", "").strip()
        d_text = options.get("D", "").strip()

        q_lines = [
            "------------------------------",
            nested_question_prompt,
            "----------",
            f"Question: {question}",
            f"A: {a_text}",
            f"B: {b_text}",
            f"C: {c_text}",
            f"D: {d_text}",
            "----------",
            "Confidence options:",
        ]

        for letter in "ABCDEFGH":
            q_lines.append(f"{letter}: {confidence_bins[letter]}")

        q_lines.extend([
            "------------------------------",
            "Your choice (A, B, C, D, E, F, G, or H):"
        ])

        # Build full user content
        user_content = setup_prompt + "\n\n" + "\n".join(q_lines)
        
        # Apply chat template
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(prompt)

    return prompts


def build_other_confidence_prompts(batch):
    """
    Build plain-text 'other-confidence' prompts (college-educated predictions).
    No ChatML tokens. Next token is directly in A–H.
    """

    setup_prompt = (
        "I want your help calibrating question difficulty. I'm going to show you a "
        "multiple-choice question, and I want you to tell me approximately what "
        "percentage of college-educated people you think would get it right. Respond "
        "only with the letter corresponding to the percentage range you choose; do NOT "
        "output any other text."
    )

    confidence_bins = {
        "A": "<5%",
        "B": "5-10%",
        "C": "10-20%",
        "D": "20-40%",
        "E": "40-60%",
        "F": "60-80%",
        "G": "80-90%",
        "H": ">90%",
    }

    nested_question_prompt = (
        "What percentage of college-educated people would get this question right?"
    )

    prompts = []
    for row in batch:
        question = row["question"].strip()
        options = row.get("options", {})

        a_text = options.get("A", "").strip()
        b_text = options.get("B", "").strip()
        c_text = options.get("C", "").strip()
        d_text = options.get("D", "").strip()

        q_lines = [
            setup_prompt,
            "",
            "------------------------------",
            nested_question_prompt,
            "----------",
            f"Question: {question}",
            f"A: {a_text}",
            f"B: {b_text}",
            f"C: {c_text}",
            f"D: {d_text}",
            "----------",
            "Confidence options:",
        ]

        for letter in "ABCDEFGH":
            q_lines.append(f"{letter}: {confidence_bins[letter]}")

        q_lines.extend([
            "------------------------------",
            "Your choice (A, B, C, D, E, F, G, or H):"
        ])

        prompt = "\n".join(q_lines) + " "
        prompts.append(prompt)

    return prompts



# ============================================================
# Dataset: no logprobs needed, only the raw MCQ fields
# ============================================================




# def get_single_token_id(tokenizer, letter: str, context: str = None) -> int:
#     """
#     Find a single-token representation for a letter.
#     Try ' A' first (common for LLaMA), then 'A'.
    
#     If context is provided, tokenize in that context to ensure we get the right token ID
#     that matches how the model will see it during inference.
#     """
#     if context is not None:
#         # Tokenize the context + letter, then extract just the letter's token ID
#         full_text = context + letter
#         ids = tokenizer.encode(full_text, add_special_tokens=False)
#         context_ids = tokenizer.encode(context, add_special_tokens=False)
#         if len(ids) == len(context_ids) + 1:
#             # The letter was tokenized as a single token
#             return ids[-1]
#         # Fallback to standard method if context doesn't help
    
#     # Try with leading space (common SPM pattern)
#     ids = tokenizer.encode(" " + letter, add_special_tokens=False)
#     if len(ids) == 1:
#         return ids[0]
#     # Fallback: bare letter
#     ids = tokenizer.encode(letter, add_special_tokens=False)
#     if len(ids) == 1:
#         return ids[0]
#     raise ValueError(f"Could not find a single-token encoding for {letter}: got {ids}")

# def get_single_token_id(tokenizer, letter: str, context: str = None) -> int:
#     """
#     Find a single-token representation for a letter.
#     Prioritize bare letter 'A' (ID 32) over ' A' (ID 320) for Llama-3.
#     """
#     if context is not None:
#         full_text = context + letter
#         ids = tokenizer.encode(full_text, add_special_tokens=False)
#         context_ids = tokenizer.encode(context, add_special_tokens=False)
#         if len(ids) == len(context_ids) + 1:
#             return ids[-1]
    
#     # PRIORITY CHANGE: Check bare letter FIRST
#     ids = tokenizer.encode(letter, add_special_tokens=False)
#     if len(ids) == 1:
#         return ids[0]

#     # Then check with leading space
#     ids = tokenizer.encode(" " + letter, add_special_tokens=False)
#     if len(ids) == 1:
#         return ids[0]
        
#     raise ValueError(f"Could not find a single-token encoding for {letter}: got {ids}")


def get_single_token_id(tokenizer, letter: str, context: str = None) -> int:
    """
    Find a single-token representation for a letter.
    For Llama-3, ' A' (with space) is typically correct in continuation context.
    """
    # Try with leading space FIRST (correct for continuations)
    ids = tokenizer.encode(" " + letter, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]

    # Fallback: bare letter
    ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
        
    raise ValueError(f"Could not find a single-token encoding for {letter}")

def compute_ABCD_entropy(probs):
    """
    Compute entropy from probability distribution for A, B, C, D options.

    Args:
        probs: tensor, list, or dict - probabilities for A, B, C, D
               If dict, should have keys "A", "B", "C", "D"
               If list/tensor, should be in order [A, B, C, D]

    Returns:
        scalar entropy value
    """
    import torch
    
    # Handle dictionary format (keys: "A", "B", "C", "D")
    if isinstance(probs, dict):
        probs = [
            probs.get("A", 0.0),
            probs.get("B", 0.0),
            probs.get("C", 0.0),
            probs.get("D", 0.0)
        ]

    # Convert to tensor if needed
    if not isinstance(probs, torch.Tensor):
        probs = torch.tensor(probs, dtype=torch.float32)

    # Ensure probabilities sum to 1
    probs = probs / (probs.sum() + 1e-12)

    # Compute entropy (natural logs)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    return entropy


def convert_entropy_to_soft_labels(entropy, sigma=10.0):
    """
    Convert entropy value to soft 8-bin confidence distribution.
    Handles both Tensor inputs (training) and float inputs (evaluation).
    """
    # Fix: Ensure input is a tensor so we can access .device or operate on it
    if not isinstance(entropy, torch.Tensor):
        entropy = torch.tensor(entropy, dtype=torch.float32)

    # Get device from entropy tensor (defaults to cpu if created from float)
    device = entropy.device
    
    # Convert entropy to "confidence percentage"
    # confidence = (1 - H/log(4)) * 100
    confidence_percent = (1 - entropy / math.log(4)) * 100.0

    # Bin midpoints + widths
    bin_edges = torch.tensor(
        [0, 5, 10, 20, 40, 60, 80, 90, 100],
        dtype=torch.float32,
        device=device
    )
    bin_midpoints = torch.tensor(
        [2.5, 7.5, 15, 30, 50, 70, 85, 95],
        dtype=torch.float32,
        device=device
    )
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Gaussian kernel in percentage space
    # Handle broadcasting for both scalar [1] and batched [B] inputs
    if entropy.ndim > 0:
        distances = (bin_midpoints.unsqueeze(0) - confidence_percent.unsqueeze(-1)) ** 2
        weights = torch.exp(-distances / (2 * sigma * sigma)) * bin_widths.unsqueeze(0)
    else:
        # Scalar case (often hits here during simple eval loops)
        distances = (bin_midpoints - confidence_percent) ** 2
        weights = torch.exp(-distances / (2 * sigma * sigma)) * bin_widths

    # Normalize along the last dimension
    return weights / weights.sum(dim=-1, keepdim=True)


def shuffle_options_and_update_correct_letter(row):
    """
    Shuffle the options (A, B, C, D) and update the correct_letter accordingly.
    
    This ensures the correct answer isn't always in position A, preventing
    position bias in the model.
    
    Args:
        row: Dictionary with "options" (dict with keys A, B, C, D) and "correct_letter"
        
    Returns:
        row: Modified row with shuffled options and updated correct_letter
    """
    if "options" not in row or "correct_letter" not in row:
        return row
    
    options = row["options"]
    correct_letter = row["correct_letter"]
    
    # Get the correct answer text
    correct_answer_text = options.get(correct_letter, "")
    
    # Create list of (letter, text) pairs
    option_pairs = [(letter, options[letter]) for letter in "ABCD" if letter in options]
    
    # Shuffle the pairs
    random.shuffle(option_pairs)
    
    # Rebuild options dict with new letter assignments
    new_options = {}
    new_correct_letter = None
    for new_letter, (old_letter, text) in zip("ABCD", option_pairs):
        new_options[new_letter] = text
        if old_letter == correct_letter:
            new_correct_letter = new_letter
    
    # Update row
    row["options"] = new_options
    row["correct_letter"] = new_correct_letter
    
    return row


# Data loading functions moved to finetune_data_handling.py


def verify_model_answer_match(pred_probs, result_data, qid=None,
                               log_file_path=None):
    """
    Check whether the model's predicted answer matches pre-recorded answer.

    Args:
        pred_probs: tensor of shape [4] - probs for A,B,C,D in order
        result_data: dict containing "subject_answer"
        qid: question ID
        log_file_path: path for write_log()
    """
    if result_data is None:
        return

    rec_ans = result_data.get("subject_answer")
    if rec_ans is None:
        return

    # model's predicted answer letter
    pred_idx = pred_probs.argmax().item()
    pred_letter = "ABCD"[pred_idx]

    # match?
    matched = (pred_letter == rec_ans)

    if log_file_path:
        write_log(log_file_path, {
            "type": ("model_answer_match" if matched
                     else "model_answer_mismatch"),
            "qid": qid,
            "predicted_answer": pred_letter,
            "recorded_answer": rec_ans,
            "predicted_probs": pred_probs.tolist(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


# ============================================================
# Weights & Biases and HuggingFace Hub utilities
# ============================================================


def init_wandb(project, run_name=None, config=None, tags=None, notes=None,
               script_path=None):
    """
    Initialize Weights & Biases logging.

    Args:
        project: W&B project name
        run_name: W&B run name (auto-generated if None)
        config: Dictionary of configuration parameters
        tags: List of tags for the run
        notes: Notes/description for the run
        script_path: Path to training script to save for reproducibility
    """
    try:
        import wandb
        wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags if tags else None,
            notes=notes if notes else None,
        )
        if script_path:
            wandb.save(
                script_path,
                base_path=os.path.dirname(os.path.abspath(script_path))
            )
        return wandb
    except ImportError:
        print("Warning: wandb not installed, skipping W&B logging")
        return None


def log_wandb_metrics(metrics, step=None):
    """Log metrics to Weights & Biases."""
    try:
        import wandb
        wandb.log(metrics, step=step)
    except (ImportError, AttributeError):
        pass  # Silently fail if wandb not available


def log_wandb_config(updates, allow_val_change=False):
    """Update W&B config with new values.
    
    Args:
        updates: Dictionary of config values to update
        allow_val_change: If True, allows changing existing config values
    """
    try:
        import wandb
        wandb.config.update(updates, allow_val_change=allow_val_change)
    except (ImportError, AttributeError):
        pass


def log_device_info(device):
    """Log device information to W&B."""
    try:
        import wandb
        import torch
        log_wandb_config({"actual_device": device})
        if device == "cuda" and torch.cuda.is_available():
            log_wandb_config({
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_memory_gb": (
                    torch.cuda.get_device_properties(0).total_memory / 1e9
                )
            })
    except (ImportError, AttributeError):
        pass


def save_hf_checkpoint(model, tokenizer, checkpoint_repo, step,
                       private=False):
    """
    Save checkpoint to HuggingFace Hub.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        checkpoint_repo: Repository name for checkpoint
        step: Training step number
        private: Whether to make the repo private

    Returns:
        True if successful, False otherwise
    """
    try:
        checkpoint_name = f"{checkpoint_repo}-step-{step}"
        model.push_to_hub(
            checkpoint_name,
            private=private,
            commit_message=f"Checkpoint at step {step}"
        )
        tokenizer.push_to_hub(
            checkpoint_name,
            private=private,
            commit_message=f"Tokenizer checkpoint at step {step}"
        )
        log_wandb_metrics({"checkpoint/hf_repo": checkpoint_name}, step=step)
        return True
    except Exception as e:
        print(f"Warning: Failed to save checkpoint to HuggingFace Hub: {e}")
        return False


def save_model_final(model, tokenizer, output_dir, hf_repo=None,
                      hf_private=False, save_wandb_artifact=False):
    """
    Save final model locally and optionally to HuggingFace Hub.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Local directory to save model
        hf_repo: HuggingFace Hub repository name (optional)
        hf_private: Whether to make HF repo private
        save_wandb_artifact: Whether to save as W&B artifact

    Returns:
        True if successful, False otherwise
    """
    # Save locally
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Save as W&B artifact
    if save_wandb_artifact:
        try:
            import wandb
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.name}",
                type="model",
                description=(
                    f"Fine-tuned ECT model with LoRA. "
                    f"Config: {wandb.config}"
                )
            )
            artifact.add_dir(output_dir)
            wandb.log_artifact(artifact)
            print(f"Model saved as wandb artifact: {artifact.name}")
        except (ImportError, AttributeError):
            print("Warning: Could not save W&B artifact")

    # Push to HuggingFace Hub
    if hf_repo:
        try:
            model.push_to_hub(hf_repo, private=hf_private)
            tokenizer.push_to_hub(hf_repo, private=hf_private)
            log_wandb_config({"hf_repo": hf_repo})
            print(f"Model and tokenizer pushed to HuggingFace Hub: {hf_repo}")
            return True
        except Exception as e:
            print(f"Warning: Failed to push to HuggingFace Hub: {e}")
            return False

    return True


def finish_wandb():
    """Finish W&B run."""
    try:
        import wandb
        wandb.finish()
    except (ImportError, AttributeError):
        pass


def check_and_clear_gpu_memory(device, min_free_gb=5.0):
    """
    Check GPU memory status and clear cache if needed.
    
    Args:
        device: Device string ("cuda" or "cpu")
        min_free_gb: Minimum free memory in GB to warn about
        
    Returns:
        dict with memory info if CUDA, None otherwise
    """
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                return None
                
            # Clear any cached memory from previous runs
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            free_memory = total_memory - reserved
            
            total_memory_gb = total_memory / (1024**3)
            allocated_gb = allocated / (1024**3)
            reserved_gb = reserved / (1024**3)
            free_memory_gb = free_memory / (1024**3)
            
            print(f"GPU Memory Status (after cache clear):")
            print(f"  Total: {total_memory_gb:.2f} GB")
            print(f"  Allocated: {allocated_gb:.2f} GB")
            print(f"  Reserved: {reserved_gb:.2f} GB")
            print(f"  Free: {free_memory_gb:.2f} GB")
            
            # Warn if memory is low
            if free_memory_gb < min_free_gb:
                print(f"\n⚠️  WARNING: Low GPU memory ({free_memory_gb:.2f} GB free)")
                print("   Llama-3-8B needs ~16GB to load. This may cause out-of-memory errors.")
                print("\n   Solutions:")
                print("   1. Restart Python/Python process to clear reserved memory")
                print("   2. Run: python -c 'import torch; torch.cuda.empty_cache()' in another terminal")
                print("   3. Restart the vast.ai instance to fully clear GPU memory")
                print("   4. Check for zombie processes: ps aux | grep python")
            
            return {
                "total_gb": total_memory_gb,
                "allocated_gb": allocated_gb,
                "reserved_gb": reserved_gb,
                "free_gb": free_memory_gb
            }
        except Exception:
            return None
    return None


def load_model_with_error_handling(model_name, device):
    """
    Load model with memory-efficient settings and proper error handling.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on ("cuda" or "cpu")
        
    Returns:
        Loaded model
        
    Raises:
        torch.cuda.OutOfMemoryError: If GPU out of memory with helpful message
    """
    import torch
    from transformers import AutoModelForCausalLM
    
    try:
        # Use low_cpu_mem_usage to reduce peak memory during loading
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map=None
        )
        
        # Move to device after loading
        if device == "cuda":
            torch.cuda.empty_cache()
            model = model.to(device)
        else:
            model = model.to(device)
            
        return model
    except torch.cuda.OutOfMemoryError as e:
        print("\n❌ CUDA Out of Memory Error!")
        print("The GPU does not have enough free memory to load the model.")
        print("\nCurrent GPU memory status:")
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            free = total - reserved
            print(f"  Total: {total:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Free: {free:.2f} GB")
        print("\n🔍 Check what's using GPU memory:")
        print("   nvidia-smi")
        print("\n💡 Solutions (try in order):")
        print("1. Kill other processes using the GPU:")
        print("   nvidia-smi  # Find PIDs")
        print("   kill <PID>  # Kill processes")
        print("\n2. Try setting expandable_segments to reduce fragmentation:")
        print("   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("   # Then re-run your script")
        print("\n3. Restart the vast.ai instance to fully clear GPU memory")
        print("\n4. Use a GPU with more memory or reduce model size")
        raise


def setup_tokenizer(model_name):
    """
    Load and configure tokenizer for causal LM.
    
    Args:
        model_name: HuggingFace model name or path
        
    Returns:
        Configured tokenizer
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad_token if it doesn't exist (common for Llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Standard for causal LMs
    return tokenizer


# Data loading functions moved to finetune_data_handling.py


def log_prompts_and_responses(step, prompt_log_file_path, answer_logits4, conf_logits8,
                               batch, answer_prompts, confidence_prompts, soft_targets):
    """
    Log prompts and responses for first 2 steps.
    
    Args:
        step: Current training step
        prompt_log_file_path: Path to log file for prompts
        answer_logits4: Logits for MCQ answers [B, 4]
        conf_logits8: Logits for confidence bins [B, 8]
        batch: Batch of question data
        answer_prompts: List of MCQ prompts
        confidence_prompts: List of confidence prompts
        soft_targets: Soft target distributions [B, 8]
    """
    # if step is not None and step < 3 and prompt_log_file_path:
    #     # Compute probabilities and predictions for logging
    #     with torch.no_grad():
    #         # First forward pass: MCQ answer probabilities
    #         mcq_probs = torch.softmax(answer_logits4, dim=-1)  # [B, 4]
    #         mcq_predicted = mcq_probs.argmax(dim=-1)  # [B]
            
    #         # Second forward pass: Confidence bin probabilities
    #         conf_probs = torch.softmax(conf_logits8, dim=-1)  # [B, 8]
    #         conf_predicted = conf_probs.argmax(dim=-1)  # [B]
        
    #     for i in range(len(batch)):
    #         # Convert predictions to letters
    #         mcq_pred_letter = "ABCD"[mcq_predicted[i].item()]
    #         conf_pred_letter = "ABCDEFGH"[conf_predicted[i].item()]
            
    #         log_entry = {
    #             "type": "prompt_and_response_pair",
    #             "step": step,
    #             "batch_index": i,
    #             "qid": batch[i].get("qid"),
    #             "mcq_prompt": answer_prompts[i],
    #             "mcq_response": {
    #                 "logits": answer_logits4[i].cpu().tolist(),
    #                 "probabilities": mcq_probs[i].cpu().tolist(),
    #                 "predicted_answer": mcq_pred_letter,
    #                 "probabilities_dict": {
    #                     "A": float(mcq_probs[i][0].item()),
    #                     "B": float(mcq_probs[i][1].item()),
    #                     "C": float(mcq_probs[i][2].item()),
    #                     "D": float(mcq_probs[i][3].item()),
    #                 }
    #             },
    #             "confidence_prompt": confidence_prompts[i],
    #             "confidence_response": {
    #                 "logits": conf_logits8[i].cpu().tolist(),
    #                 "probabilities": conf_probs[i].cpu().tolist(),
    #                 "predicted_bin": conf_pred_letter,
    #                 "probabilities_dict": {
    #                     "A": float(conf_probs[i][0].item()),
    #                     "B": float(conf_probs[i][1].item()),
    #                     "C": float(conf_probs[i][2].item()),
    #                     "D": float(conf_probs[i][3].item()),
    #                     "E": float(conf_probs[i][4].item()),
    #                     "F": float(conf_probs[i][5].item()),
    #                     "G": float(conf_probs[i][6].item()),
    #                     "H": float(conf_probs[i][7].item()),
    #                 }
    #             },
    #             "soft_targets": soft_targets[i].cpu().tolist(),
    #             "timestamp": datetime.now(timezone.utc).isoformat()
    #         }
    #         write_log(prompt_log_file_path, log_entry)
            
    #         # Also print to console
    #         print(f"\n{'='*80}")
    #         print(f"STEP {step} | BATCH INDEX {i} | QID: {batch[i].get('qid')}")
    #         print(f"{'='*80}")
    #         print(f"\nMCQ PROMPT (First forward pass):")
    #         print(f"{'-'*80}")
    #         print(answer_prompts[i])
    #         print(f"\nMCQ RESPONSE:")
    #         print(f"{'-'*80}")
    #         print(f"  Predicted Answer: {mcq_pred_letter}")
    #         print(f"  Probabilities: A={mcq_probs[i][0]:.4f}, B={mcq_probs[i][1]:.4f}, "
    #               f"C={mcq_probs[i][2]:.4f}, D={mcq_probs[i][3]:.4f}")
    #         print(f"  Logits: {answer_logits4[i].cpu().tolist()}")
    #         print(f"\nCONFIDENCE PROMPT (Second forward pass - separate context):")
    #         print(f"{'-'*80}")
    #         print(confidence_prompts[i])
    #         print(f"\nCONFIDENCE RESPONSE:")
    #         print(f"{'-'*80}")
    #         print(f"  Predicted Bin: {conf_pred_letter}")
    #         conf_bin_labels = ["A: <5%", "B: 5-10%", "C: 10-20%", "D: 20-40%",
    #                           "E: 40-60%", "F: 60-80%", "G: 80-90%", "H: >90%"]
    #         print(f"  Probabilities:")
    #         for j, label in enumerate(conf_bin_labels):
    #             print(f"    {label}: {conf_probs[i][j]:.4f}")
    #         print(f"  Logits: {conf_logits8[i].cpu().tolist()}")
    #         print(f"\nSOFT TARGETS (Training target):")
    #         print(f"{'-'*80}")
    #         print(f"  {soft_targets[i].cpu().tolist()}")
    #         print(f"{'='*80}\n")

