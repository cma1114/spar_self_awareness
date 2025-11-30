import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # <--- ADDED FOR LORA
import random
import wandb

from finetune_utils import (
    write_log,
    # get_single_token_id,
    build_multiple_choice_question_prompts,
    build_self_confidence_prompts,
    # setup_tokenizer,
    # load_model_with_error_handling,
    # check_and_clear_gpu_memory,
    # compute_ABCD_entropy,
    shuffle_options_and_update_correct_letter,
    # log_wandb_metrics,
    # parse_letter_from_model_text,
    run_mcq_forward_pass,
    run_confidence_forward_pass,
    convert_entropy_to_soft_labels,
    prepare_model_and_tokenizer
)
from finetune_data_handling import (
    # MCQDataset,
    # normalize_text,
    # validate_and_load_dataset,
    verify_and_resolve_options,
    write_jsonl,
)
from finetune_data_handling import collate_fn


# ============================================================
# Validation and Test Evaluation Functions
# ============================================================


def run_evaluation(
    model,
    tokenizer,
    val_dataset,
    device,
    args,
    step_name=None,
    num_samples=None,
    log_file_path=None,
    step=None,
):
    """
    Evaluation loop:

    For each sample:
        1. Run MCQ pass (extract predicted letter + entropy)
        2. Run confidence pass (A–H distribution)
        3. Compute soft targets from entropy
        4. Compute loss
        5. Log per-sample results

    After loop:
        - Compute MCQ accuracy
        - Compute average entropy
        - Compute average expected confidence
        - Compute average loss
        - Compute model answer distribution
        - Compute correct answer distribution
    """

    model.eval()

    # ===========================================================
    # Select samples
    # ===========================================================
    if num_samples is not None:
        idxs = np.random.choice(len(val_dataset), size=num_samples, replace=False)
    else:
        idxs = np.arange(len(val_dataset))

    # At the start of run_evaluation()
    print("DEBUG: First question:")
    print(f"Question: {val_dataset[0]['question'][:100]}")
    print(f"Options: {val_dataset[0].get('options', {})}")
    print(f"Correct: {val_dataset[0].get('correct_letter')}")
    print(f"Shuffle setting: {args.shuffle_options}")

    # ----- Accumulators -----
    correctness_flags = []
    entropy_values = []
    expected_conf_values = []
    loss_values = []
    predicted_letters = []
    correct_letters = []

    # ==================== MAIN LOOP ==========================
    for i in idxs:
        batch = val_dataset[i:i+1]    # single-sample batch (list of 1 dict)

        # 0. Resolve options and assign correct_letter
        resolved_row, options = verify_and_resolve_options(
            batch[0],
            mcq_results_lookup=None,   # or whatever you use here
            log_file_path=None
        )
        batch[0]["options"] = options

        # Optional shuffle
        if args.shuffle_options:
            shuffle_options_and_update_correct_letter(batch[0])


        # ==================================================
        # 1. MCQ pass
        # ==================================================
        mcq_prompts = build_multiple_choice_question_prompts(batch, tokenizer)

        mcq_out = run_mcq_forward_pass(
            model=model,
            tokenizer=tokenizer,
            prompts=mcq_prompts,
            device=device,
            temperature=0.0,
        )

        predicted_answer_letter = mcq_out["pred_letters"][0]
        entropy_value = mcq_out["entropy"][0].item()

        correct_answer_letter = batch[0]["correct_letter"]

        predicted_letters.append(predicted_answer_letter)
        correct_letters.append(correct_answer_letter)

        correctness_flags.append(
            1.0 if predicted_answer_letter == correct_answer_letter else 0.0
        )
        entropy_values.append(entropy_value)

        # ==================================================
        # 2. Soft targets
        # ==================================================
        soft_targets = convert_entropy_to_soft_labels(entropy_value).to(device)

        # ==================================================
        # 3. Confidence pass
        # ==================================================
        conf_prompts = build_self_confidence_prompts(batch, tokenizer)

        conf_out = run_confidence_forward_pass(
            model=model,
            tokenizer=tokenizer,
            prompts=conf_prompts,
            device=device,
            temperature=args.temperature,
        )

        logits8 = conf_out["logits8"][0]
        expected_confidence_value = conf_out["expected_conf"][0].item()

        expected_conf_values.append(expected_confidence_value)

        # ==================================================
        # 4. Loss
        # ==================================================
        log_probs = torch.log_softmax(logits8, dim=-1)
        loss_value = -(soft_targets * log_probs).sum().item()
        loss_values.append(loss_value)

        # ==================================================
        # 5. Optional per-sample logging
        # ==================================================
        if log_file_path is not None:
            write_jsonl(log_file_path, {
                "type": "eval_sample",
                "qid": batch[0].get("qid"),
                "question": batch[0]["question"],
                "model_answer": predicted_answer_letter,
                "correct_answer": correct_answer_letter,
                "entropy": entropy_value,
                "expected_confidence": expected_confidence_value,
                "loss": loss_value,
            })



    # ===========================================================
    # DISTRIBUTION STATS (A–D)
    # ===========================================================

    def count_dist(values):
        return {letter: values.count(letter) for letter in "ABCD"}

    pred_dist = count_dist(predicted_letters)
    gold_dist = count_dist(correct_letters)

    n = len(predicted_letters)

    pred_dist_pct = {k: (v / n) * 100.0 for k, v in pred_dist.items()}
    gold_dist_pct = {k: (v / n) * 100.0 for k, v in gold_dist.items()}

    # Pretty print
    print("\n============================================================")
    print(f"{step_name.upper()} — MCQ Answer Distributions")
    print("============================================================")
    print("Correct (Ground Truth) Distribution:")
    for k in "ABCD":
        print(f"  {k}: {gold_dist[k]:4d}  ({gold_dist_pct[k]:6.2f}%)")

    print("\nModel Prediction Distribution:")
    for k in "ABCD":
        print(f"  {k}: {pred_dist[k]:4d}  ({pred_dist_pct[k]:6.2f}%)")

    # ===========================================================
    # FINAL METRICS
    # ===========================================================
    results = {
        "mcq_accuracy": float(np.mean(correctness_flags)),
        "avg_entropy": float(np.mean(entropy_values)),
        "avg_confidence": float(np.mean(expected_conf_values)),
        "avg_loss": float(np.mean(loss_values)),
        "n_samples": n,
        "correct_answer_distribution_raw": gold_dist,
        "correct_answer_distribution_pct": gold_dist_pct,
        "predicted_answer_distribution_raw": pred_dist,
        "predicted_answer_distribution_pct": pred_dist_pct,
    }

    # Compute additional metrics for wandb
    correctness_arr = np.array(correctness_flags)
    entropy_arr = np.array(entropy_values)
    conf_arr = np.array(expected_conf_values)
    
    # Standard deviation of confidence (mode collapse check)
    std_conf = float(np.std(conf_arr)) if len(conf_arr) > 1 else 0.0
    std_entropy = float(np.std(entropy_arr)) if len(entropy_arr) > 1 else 0.0
    
    # Alignment: correlation between entropy and confidence (should be negative)
    alignment_corr = 0.0
    if len(conf_arr) > 1 and std_conf > 0.001:
        try:
            alignment_corr, _ = pearsonr(entropy_arr, conf_arr)
            alignment_corr = float(alignment_corr)
        except Exception:
            alignment_corr = 0.0
    
    # Calibration: correlation between confidence and correctness
    calibration_corr = 0.0
    if len(conf_arr) > 1 and std_conf > 0.001:
        try:
            calibration_corr, _ = pearsonr(conf_arr, correctness_arr)
            calibration_corr = float(calibration_corr)
        except Exception:
            calibration_corr = 0.0

    # Log the summary as one blob
    if log_file_path is not None:
        write_jsonl(log_file_path, {
            "type": "eval_summary",
            **results
        })

    # ===========================================================
    # WANDB LOGGING
    # ===========================================================
    if args.save_wandb_artifact:
        try:
            
            prefix = "val" 
            
            wandb_metrics = {
                f"{prefix}/accuracy": results["mcq_accuracy"],
                f"{prefix}/loss": results["avg_loss"],
                f"{prefix}/entropy": results["avg_entropy"],
                f"{prefix}/confidence": results["avg_confidence"],
                f"{prefix}/std_confidence": std_conf,
                f"{prefix}/std_entropy": std_entropy,
                f"{prefix}/alignment_corr": alignment_corr,
                f"{prefix}/calibration_corr": calibration_corr,
                f"{prefix}/n_samples": n,
            }
            
            # Add answer distribution percentages
            for letter in "ABCD":
                wandb_metrics[f"{prefix}/pred_dist_{letter}_pct"] = pred_dist_pct[letter]
                wandb_metrics[f"{prefix}/correct_dist_{letter}_pct"] = gold_dist_pct[letter]
            
            # Add step if provided
            if step is not None:
                wandb_metrics["step"] = step
            
            wandb.log(wandb_metrics)
        except (ImportError, AttributeError):
            pass  # Silently fail if wandb not available

    return results


def compute_calibration_metrics(correctness, confidence, n_bins=10):
    """
    Compute calibration metrics: ECE, Brier score, and decomposition.
    """
    # Align and drop NaNs
    data = pd.DataFrame({
        'correct': correctness,
        'prob': confidence
    }).dropna()
    
    if len(data) == 0:
        return {
            'ece': np.nan,
            'brier': np.nan,
            'reliability': np.nan,
            'resolution': np.nan,
            'uncertainty': np.nan,
        
        }
    
    n_samples = len(data)
    base_rate = data['correct'].mean()
    
    # 1. Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (data['prob'] > bin_lower) & (data['prob'] <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            bin_acc = data.loc[in_bin, 'correct'].mean()
            bin_conf = data.loc[in_bin, 'prob'].mean()
            ece += prop_in_bin * abs(bin_acc - bin_conf)
    
    # 2. Brier Score and Decomposition
    brier = ((data['prob'] - data['correct']) ** 2).mean()
    
    # Reliability (calibration)
    reliability = 0.0
    resolution = 0.0
    
    for i in range(n_bins):
        bin_mask = (data['prob'] >= bin_boundaries[i]) & (data['prob'] < bin_boundaries[i+1])
        n_bin = bin_mask.sum()
        
        if n_bin > 0:
            bin_prob = data.loc[bin_mask, 'prob'].mean()
            bin_freq = data.loc[bin_mask, 'correct'].mean()
            bin_weight = n_bin / len(data)
            
            reliability += bin_weight * (bin_prob - bin_freq) ** 2
            resolution += bin_weight * (bin_freq - base_rate) ** 2
    
    uncertainty = base_rate * (1 - base_rate)

    
    return {
        'ece': float(ece),
        'brier': float(brier),
        'reliability': float(reliability),
        'resolution': float(resolution),
        'uncertainty': float(uncertainty),

        'n_samples': n_samples,
    }


# def compute_metrics_for_wandb(all_correct, all_entropies, all_verbal_conf):
#     """
#     Aggregates validation results into metrics for WandB.
    
#     Args:
#         all_correct: list of 1s (correct) and 0s (incorrect)
#         all_entropies: list of internal entropy values
#         all_verbal_conf: list of predicted verbal confidence scores (0-100)
        
#     Returns:
#         dict: Metrics ready for wandb.log()
#     """
#     # Convert to numpy for easy math
#     correct_arr = np.array(all_correct)
#     entropies_arr = np.array(all_entropies)
#     conf_arr = np.array(all_verbal_conf)
    
#     # 1. Capability: Is it still answering questions correctly?

    
#     # 2. Mode Collapse: Is it outputting the same confidence everywhere?
#     avg_conf = np.mean(conf_arr)
#     std_conf = np.std(conf_arr)
    
#     # 3. Alignment: Does uncertainty (entropy) match verbal report?
#     # We expect NEGATIVE correlation (High Entropy = Low Confidence)
#     try:
#         if len(conf_arr) > 1 and std_conf > 0.001:
#             align_corr, _ = pearsonr(entropies_arr, conf_arr)
#         else:
#             align_corr = 0.0
#     except Exception:
#         align_corr = 0.0
        
#     # 4. Calibration: Is the confidence useful?
#     try:
#         if len(conf_arr) > 1 and std_conf > 0.001:
#             calib_corr, _ = pearsonr(conf_arr, correct_arr)
#         else:
#             calib_corr = 0.0
#     except Exception:
#         calib_corr = 0.0

#     return {
   
#         "val/avg_verbal_conf": avg_conf,
#         "val/std_verbal_conf": std_conf,  # < 2.0 = Collapse
#         "val/alignment_corr": align_corr, # The most important metric
#         "val/calibration_corr": calib_corr
#     }


def log_answer_distributions(log_file_path, step_type, step_number, 
                             predicted_letter_counts, correct_letter_counts, 
                             total_questions, accuracy=None, avg_entropy=None,
                             answer_variety=None, answer_entropy_std=None):
    """
    Log answer distribution information to a dedicated log file.
    
    Args:
        log_file_path: Path to the answer distributions log file
        step_type: "val" or "test"
        step_number: Step number (for validation) or None (for test)
        predicted_letter_counts: dict with counts for A, B, C, D
        correct_letter_counts: dict with counts for A, B, C, D
        total_questions: Total number of questions
        accuracy: Optional accuracy value
        avg_entropy: Optional average entropy value
        answer_variety: Optional answer variety score (0 = always same, 1 = 25% each)
        answer_entropy_std: Optional standard deviation of answer entropy
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Calculate percentages
    predicted_percentages = {
        letter: (count / total_questions * 100) if total_questions > 0 else 0.0
        for letter, count in predicted_letter_counts.items()
    }
    correct_percentages = {
        letter: (count / total_questions * 100) if total_questions > 0 else 0.0
        for letter, count in correct_letter_counts.items()
    }
    
    log_entry = {
        "type": "answer_distribution",
        "timestamp": timestamp,
        "step_type": step_type,  # "val" or "test"
        "step_number": step_number,  # None for test
        "total_questions": total_questions,
        "accuracy": accuracy,
        "avg_entropy": avg_entropy,
        "answer_entropy_std": answer_entropy_std,
        "answer_variety": answer_variety,
        "predicted_letter_distribution": {
            letter: {
                "count": predicted_letter_counts.get(letter, 0),
                "percentage": predicted_percentages.get(letter, 0.0)
            }
            for letter in "ABCD"
        },
        "correct_letter_distribution": {
            letter: {
                "count": correct_letter_counts.get(letter, 0),
                "percentage": correct_percentages.get(letter, 0.0)
            }
            for letter in "ABCD"
        }
    }
    
    write_log(log_file_path, log_entry)






# ==============================================
# MAYBE DELETE EVERYTHING BELOW
# ==============================================




# def run_inference(model, tokenizer, dataset, device="cuda", batch_size=4):
#     """Run inference to get verbalized confidence scores."""
#     model.eval()
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
#     all_predictions = []
#     all_verbal_confidences = [] # Renamed for clarity
#     all_correctness = []
#     all_qids = []
#     all_entropies = []
    
#     print(f"Running inference on {len(dataset)} samples...")
    
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(dataloader):
#             if batch_idx % 10 == 0:
#                 print(f"  Processing batch {batch_idx + 1}/{len(dataloader)}")
            
#             # --- PASS 1: ANSWER ---
#             answer_prompts = build_multiple_choice_question_prompts(batch)
#             enc = tokenizer(answer_prompts, return_tensors="pt", padding=True).to(device)
            
#             try:
#                 out = model(**enc, use_cache=False)
#             except TypeError:
#                 out = model(**enc)
            
#             final_logits = out.logits[:, -1, :]
#             abcd_ids = torch.tensor(
#                 [get_single_token_id(tokenizer, c) for c in "ABCD"],
#                 device=device,
#                 dtype=torch.long
#             )
#             answer_logits4 = final_logits[:, abcd_ids]
            
#             # Answer Probs & Entropy
#             answer_probs = torch.softmax(answer_logits4, dim=-1)
#             predicted_indices = answer_probs.argmax(dim=-1)
#             predicted_letters = ["ABCD"[idx.item()] for idx in predicted_indices]
            
#             entropies = -(answer_probs * torch.log(answer_probs + 1e-12)).sum(dim=-1)
            
#             # --- PASS 2: CONFIDENCE ---
#             confidence_prompts = build_self_confidence_prompts(batch)
#             enc2 = tokenizer(confidence_prompts, return_tensors="pt", padding=True).to(device)
            
#             try:
#                 out2 = model(**enc2, use_cache=False)
#             except TypeError:
#                 out2 = model(**enc2)
            
#             final_logits2 = out2.logits[:, -1, :]
#             bins_ids = torch.tensor(
#                 [get_single_token_id(tokenizer, c) for c in "ABCDEFGH"],
#                 device=device,
#                 dtype=torch.long
#             )
#             conf_logits8 = final_logits2[:, bins_ids]
#             conf_probs = torch.softmax(conf_logits8, dim=-1)
            
#             # Map confidence bins to percentages (midpoints)
#             bin_midpoints = np.array([2.5, 7.5, 15, 30, 50, 70, 85, 95]) / 100.0
#             # Expected confidence (Verbal)
#             expected_conf = (conf_probs * torch.tensor(bin_midpoints, device=device)).sum(dim=-1)
            
#             for i, row in enumerate(batch):
#                 correct_letter = row["correct_letter"]
#                 predicted_letter = predicted_letters[i]
#                 is_correct = 1 if predicted_letter == correct_letter else 0
                
#                 # CRITICAL CHANGE: Use VERBAL expected confidence, not internal softmax max
#                 verbal_confidence = expected_conf[i].item()
#                 entropy = entropies[i].item()
                
#                 all_predictions.append(predicted_letter)
#                 all_verbal_confidences.append(verbal_confidence)
#                 all_correctness.append(is_correct)
#                 all_qids.append(row.get("qid", f"batch_{batch_idx}_item_{i}"))
#                 all_entropies.append(entropy)
    
#     return {
#         'predictions': all_predictions,
#         'confidences': np.array(all_verbal_confidences), # Now this is verbal confidence
#         'correctness': np.array(all_correctness),
#         'qids': all_qids,
#         'entropies': np.array(all_entropies),
#     }


# def run_diagnostics(base_model, data_path, checkpoint_path=None, device="cuda", 
#                     batch_size=4, log_file="diagnostics_log.jsonl", output_csv=None):
    
#     print("=" * 80)
#     print("Running Model Diagnostics")
#     print("=" * 80)
#     print(f"Base Model: {base_model}")
#     print(f"Checkpoint: {checkpoint_path}")
#     print(f"Dataset: {data_path}")
#     print()
    
#     check_and_clear_gpu_memory(device)
    
#     print("Loading tokenizer...")
#     tokenizer = setup_tokenizer(base_model)
    
#     print("Loading base model...")
#     model = load_model_with_error_handling(base_model, device)
    
#     if checkpoint_path:
#         print(f"Loading LoRA adapter from {checkpoint_path}...")
#         model = PeftModel.from_pretrained(model, checkpoint_path)
#         model = model.merge_and_unload() # Merge for efficiency
#         print("Adapter loaded and merged.")
    
#     if model.config.pad_token_id is None:
#         model.config.pad_token_id = tokenizer.pad_token_id
    
#     dataset = MCQDataset(data_path)
#     print(f"Loaded {len(dataset)} samples")
    
#     results = run_inference(model, tokenizer, dataset, device=device, batch_size=batch_size)
    
#     print("\nComputing metrics...")
#     calibration_metrics = compute_calibration_metrics(
#         results['correctness'],
#         results['confidences']
#     )
    
#     # --- METRICS CALCULATIONS ---
    
#     # 1. Mode Collapse (Std Dev of Verbal Confidence)
#     std_verbal_conf = np.std(results['confidences'])
#     avg_verbal_conf = np.mean(results['confidences'])
    
#     # 2. Alignment (Entropy vs Verbal Confidence)
#     # Ideally NEGATIVE correlation (High Entropy = Low Confidence)
#     try:
#         corr_align_p, p_align_p = pearsonr(results['entropies'], results['confidences'])
#     except:
#         corr_align_p, p_align_p = np.nan, np.nan
        
#     # 3. Usefulness (Confidence vs Correctness)
#     try:
#         corr_calib_p, p_calib_p = pearsonr(results['confidences'], results['correctness'])
#     except:
#         corr_calib_p, p_calib_p = np.nan, np.nan

#     # Print results
#     print("\n" + "=" * 80)
#     print("DIAGNOSTICS RESULTS")
#     print("=" * 80)
  
#     print("-" * 40)
#     print("MODE COLLAPSE CHECK:")
#     print(f"Avg Verbal Conf:    {avg_verbal_conf:.4f}")
#     print(f"Std Verbal Conf:    {std_verbal_conf:.4f}  (If < 0.02, model is likely collapsed)")
#     print("-" * 40)
#     print("ALIGNMENT CHECK (Metacognition):")
#     print(f"Corr(Entropy, Conf): {corr_align_p:.4f}  (Target: Negative, e.g., -0.6)")
#     print("-" * 40)
#     print("CALIBRATION CHECK:")
#     print(f"ECE:                {calibration_metrics['ece']:.4f}")
#     print(f"Brier Score:        {calibration_metrics['brier']:.4f}")
#     print(f"Corr(Conf, Correct): {corr_calib_p:.4f}")
#     print("=" * 80)
    
#     # Save raw CSV
#     if output_csv:
#         df = pd.DataFrame({
#             'qid': results['qids'],
#             'prediction': results['predictions'],
#             'verbal_confidence': results['confidences'],
#             'correctness': results['correctness'],
#             'internal_entropy': results['entropies'],
#         })
#         df.to_csv(output_csv, index=False)
#         print(f"Saved raw data to {output_csv}")
    
#     # Log entry
#     timestamp = datetime.now(timezone.utc).isoformat()
#     log_entry = {
#         "type": "diagnostics_summary",
#         "timestamp": timestamp,
#         "base_model": base_model,
#         "checkpoint": checkpoint_path,
#         "dataset": data_path,
#         "metrics": {

#             "ece": calibration_metrics['ece'],
#             "brier": calibration_metrics['brier'],
#             "std_verbal_conf": float(std_verbal_conf),
#             "avg_verbal_conf": float(avg_verbal_conf),
#             "alignment_entropy_conf_pearson": float(corr_align_p) if not np.isnan(corr_align_p) else None,
#             "calibration_conf_correct_pearson": float(corr_calib_p) if not np.isnan(corr_calib_p) else None,
#         }
#     }
    
#     if log_file:
#         write_log(log_file, log_entry)
#         print(f"Logged summary to {log_file}")



# def assess_mcq_accuracy(
#     model,
#     tokenizer,
#     val_dataset,
#     device="cuda",
#     validation_step=0,
#     log_file_path=None,
#     num_questions=500,
#     temperature=0.0,
#     seed=None,
#     shuffle_options=True,
# ):
#     """
#     Assess model accuracy on multiple choice questions from validation set.
    
#     Randomly samples questions from validation dataset, runs inference,
#     and logs results to file and W&B.
    
#     Args:
#         model: Language model
#         tokenizer: Tokenizer
#         val_dataset: MCQDataset instance
#         device: Device to run on
#         validation_step: Current validation step number (e.g., 100, 200, etc.)
#         log_file_path: Path to log file for detailed question-level logs
#         num_questions: Number of random questions to sample (default: 500)
#         temperature: Temperature for sampling predictions (0.0 = deterministic)
#         seed: Random seed for sampling (None = use current random state)
        
#     Returns:
#         dict with keys:
#             - "accuracy": average accuracy (0-1)
#             - "avg_entropy": average entropy across all questions
#     """
#     import random
#     import torch
    
#     model.eval()
    
#     # Check if we should sample or use all questions
#     dataset_size = len(val_dataset)
#     num_questions = min(num_questions, dataset_size)
    
#     # If num_questions equals dataset_size, use all questions (no sampling needed)
#     # This happens when a pre-sampled subset is passed
#     if num_questions == dataset_size:
#         # Use all questions in order (no random sampling)
#         sampled_indices = list(range(dataset_size))
#         print(f"Assessing MCQ accuracy on {num_questions} questions...")
#     else:
#         # Set seed if provided (for reproducibility)
#         if seed is not None:
#             random.seed(seed)
#         # Randomly sample questions from validation dataset
#         sampled_indices = random.sample(range(dataset_size), num_questions)
#         print(f"Assessing MCQ accuracy on {num_questions} random questions from validation set...")
    
#     all_correct = []
#     all_entropies = []
#     all_confidence_predictions = []  # For confidence assessment
#     predicted_letter_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
#     correct_letter_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    
#     # DIAGNOSTIC: Track logit statistics to detect token bias
#     logit_sums = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
#     logit_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    
#     # VERIFICATION: Check that MCQ and confidence tokens are what we expect
#     # (They should overlap for A-D, but we need to ensure we're using the right logits)
#     mcq_tokens = {letter: get_single_token_id(tokenizer, letter) for letter in "ABCD"}
#     conf_tokens = {letter: get_single_token_id(tokenizer, letter) for letter in "ABCDEFGH"}
    
#     # Verify MCQ tokens A-D match confidence tokens A-D (expected behavior)
#     tokens_match = all(mcq_tokens[letter] == conf_tokens[letter] for letter in "ABCD")
#     if not tokens_match:
#         print(f"⚠️  WARNING: MCQ tokens A-D don't match confidence tokens A-D!")
#         print(f"   MCQ tokens: {mcq_tokens}")
#         conf_tokens_abcd = {k: v for k, v in list(conf_tokens.items())[:4]}
#         print(f"   Conf tokens A-D: {conf_tokens_abcd}")
#     else:
#         if validation_step == 0 or validation_step % 100 == 0:  # Only print occasionally
#             print(f"✓ Verified: MCQ tokens (A-D) match confidence tokens (A-D) as expected")
#             print(f"   MCQ/Conf tokens A-D: {mcq_tokens}")
#             conf_tokens_efgh = {'E': conf_tokens['E'], 'F': conf_tokens['F'], 'G': conf_tokens['G'], 'H': conf_tokens['H']}
#             print(f"   Confidence tokens E-H: {conf_tokens_efgh}")
    
#     # Process in batches for efficiency
#     batch_size = 4
#     batch = []
#     batch_indices = []
    
#     with torch.no_grad():
#         for idx, sample_idx in enumerate(sampled_indices):
#             # Get question from dataset
#             row = val_dataset[sample_idx]
#             batch.append(row)
#             batch_indices.append(sample_idx)
            
#             # Process batch when full or at end
#             if len(batch) == batch_size or idx == len(sampled_indices) - 1:
#                 # Shuffle options to prevent position bias (if enabled)
#                 if shuffle_options:
#                     for row in batch:
#                         shuffle_options_and_update_correct_letter(row)
                
#                 # Build prompts using the utility function
#                 answer_prompts = build_multiple_choice_question_prompts(batch)
                
#                 # Tokenize and run inference
#                 enc = tokenizer(answer_prompts, return_tensors="pt", padding=True).to(device)
                
#                 try:
#                     out = model(**enc, use_cache=False)
#                 except TypeError:
#                     out = model(**enc)
                
#                 # Extract logits for A, B, C, D (MCQ answers only)
#                 # CRITICAL: We use answer_logits4 (4 logits) for MCQ accuracy, NOT conf_logits8 (8 logits)
#                 final_logits = out.logits[:, -1, :]
#                 abcd_ids = torch.tensor(
#                     [mcq_tokens[c] for c in "ABCD"],  # Use pre-computed tokens
#                     device=device,
#                     dtype=torch.long
#                 )
#                 answer_logits4 = final_logits[:, abcd_ids]  # [B, 4] - MCQ logits only
                
#                 # DIAGNOSTIC: Track raw logit values to detect bias
#                 # Accumulate logits across all batches
#                 batch_logits = answer_logits4.mean(dim=0).cpu().numpy()
#                 for i, letter in enumerate("ABCD"):
#                     logit_sums[letter] += batch_logits[i]
#                     logit_counts[letter] += 1
                
#                 # Compute probabilities and predictions for MCQ (with temperature)
#                 if temperature > 0:
#                     scaled_logits = answer_logits4 / temperature
#                     answer_probs = torch.softmax(scaled_logits, dim=-1)
#                     # Sample from the distribution
#                     predicted_indices = torch.multinomial(answer_probs, num_samples=1).squeeze(-1)
#                 else:
#                     answer_probs = torch.softmax(answer_logits4, dim=-1)
#                     # Deterministic: use argmax
#                     predicted_indices = answer_probs.argmax(dim=-1)
                
#                 # Now run confidence assessment on the same batch
#                 confidence_prompts = build_self_confidence_prompts(batch, tokenizer)
#                 enc2 = tokenizer(confidence_prompts, return_tensors="pt", padding=True).to(device)
                
#                 try:
#                     out2 = model(**enc2, use_cache=False)
#                 except TypeError:
#                     out2 = model(**enc2)
                
#                 final_logits2 = out2.logits[:, -1, :]
#                 bins_ids = torch.tensor(
#                     [conf_tokens[c] for c in "ABCDEFGH"],  # Use pre-computed tokens
#                     device=device,
#                     dtype=torch.long
#                 )
#                 conf_logits8 = final_logits2[:, bins_ids]  # [B, 8] - Confidence logits (separate from MCQ!)
                
#                 # Compute confidence probabilities (with temperature if > 0)
#                 if temperature > 0:
#                     scaled_conf_logits = conf_logits8 / temperature
#                     conf_probs = torch.softmax(scaled_conf_logits, dim=-1)
#                 else:
#                     conf_probs = torch.softmax(conf_logits8, dim=-1)
                
#                 # Map confidence bins to percentages (midpoints)
#                 bin_midpoints = np.array([2.5, 7.5, 15, 30, 50, 70, 85, 95]) / 100.0
#                 # Expected confidence (Verbal)
#                 expected_conf = (conf_probs * torch.tensor(bin_midpoints, device=device)).sum(dim=-1)
#                 predicted_conf_bins = conf_probs.argmax(dim=-1)
                
#                 # Process each question in batch - compute and log both MCQ and confidence
#                 for i, row in enumerate(batch):
#                     # === MCQ Assessment ===
#                     # Get predicted answer letter from MCQ logits (A-D only, NOT confidence A-H)
#                     # CRITICAL: predicted_indices comes from answer_probs which comes from answer_logits4 (4 logits)
#                     # We are NOT using conf_logits8 (8 logits) for accuracy calculation
#                     pred_idx = predicted_indices[i].item()
#                     predicted_letter = "ABCD"[pred_idx]
                    
#                     # CRITICAL: Ensure we're using MCQ answer (A-D), not confidence prediction (A-H)
#                     assert predicted_letter in "ABCD", f"MCQ prediction must be A-D, got {predicted_letter}"
#                     assert pred_idx < 4, f"MCQ prediction index must be 0-3 (A-D), got {pred_idx}"
                    
#                     # Get the exact model output: apply same temperature logic to full vocab logits
#                     if temperature > 0:
#                         scaled_full_logits = final_logits[i] / temperature
#                         full_probs = torch.softmax(scaled_full_logits, dim=-1)
#                         mcq_response_token_id = torch.multinomial(full_probs, num_samples=1).item()
#                     else:
#                         mcq_response_token_id = final_logits[i].argmax().item()
#                     mcq_response_logit = final_logits[i][mcq_response_token_id].item()
#                     mcq_response_text = tokenizer.decode([mcq_response_token_id], skip_special_tokens=False)
                    
#                     # Get correct answer letter
#                     correct_letter = row["correct_letter"]
#                     assert correct_letter in "ABCD", f"Correct answer must be A-D, got {correct_letter}"
                    
#                     # Track letter distributions for debugging (MCQ answers A-D only)
#                     predicted_letter_counts[predicted_letter] = predicted_letter_counts.get(predicted_letter, 0) + 1
#                     if correct_letter:
#                         correct_letter_counts[correct_letter] = correct_letter_counts.get(correct_letter, 0) + 1
                    
#                     # Get the actual text for both answers from options
#                     options = row.get("options", {})
#                     predicted_answer_text = options.get(predicted_letter, "")
#                     correct_answer_text = options.get(correct_letter, "")
                    
#                     # Compare normalized text instead of just letters
#                     # This ensures we catch any issues with answer matching
#                     predicted_normalized = normalize_text(predicted_answer_text)
#                     correct_normalized = normalize_text(correct_answer_text)
                    
#                     # Check if correct by comparing normalized text
#                     is_correct_by_text = 1 if predicted_normalized == correct_normalized else 0
                    
#                     # Also check by letter for logging/debugging
#                     is_correct_by_letter = 1 if predicted_letter == correct_letter else 0
                    
#                     # Use letter-based comparison for accuracy calculation (matches capabilities_test.py)
#                     # This ensures consistency: capabilities_test.py uses subject_decision == question["correct_answer"]
#                     all_correct.append(is_correct_by_letter)
                    
#                     # Debug: Warn if text and letter comparisons don't match
#                     if is_correct_by_text != is_correct_by_letter:
#                         print(f"WARNING: Mismatch for qid {row.get('qid')}: text_match={is_correct_by_text}, letter_match={is_correct_by_letter}")
#                         print(f"  Predicted: {predicted_letter}='{predicted_answer_text}' (norm: '{predicted_normalized}')")
#                         print(f"  Correct: {correct_letter}='{correct_answer_text}' (norm: '{correct_normalized}')")
#                         print(f"  Note: Using letter_match for accuracy (matching capabilities_test.py behavior)")
                    
#                     # Calculate entropy for this answer distribution
#                     probs_for_entropy = answer_probs[i].cpu()
#                     entropy = compute_ABCD_entropy(probs_for_entropy).item()
#                     all_entropies.append(entropy)
                    
#                     # === Confidence Assessment ===
#                     # NOTE: This is separate from MCQ assessment above
#                     # Confidence uses A-H scale, but we do NOT use this for accuracy calculation
#                     conf_pred_letter = "ABCDEFGH"[predicted_conf_bins[i].item()]
#                     verbal_confidence = expected_conf[i].item()
#                     # Compute confidence entropy (entropy of confidence distribution)
#                     conf_entropy = -(conf_probs[i] * torch.log(conf_probs[i] + 1e-12)).sum().item()
                    
#                     # Get the exact model output: apply same temperature logic to full vocab logits
#                     if temperature > 0:
#                         scaled_full_logits2 = final_logits2[i] / temperature
#                         full_probs2 = torch.softmax(scaled_full_logits2, dim=-1)
#                         conf_response_token_id = torch.multinomial(full_probs2, num_samples=1).item()
#                     else:
#                         conf_response_token_id = final_logits2[i].argmax().item()
#                     conf_response_logit = final_logits2[i][conf_response_token_id].item()
#                     conf_response_text = tokenizer.decode([conf_response_token_id], skip_special_tokens=False)
                    
#                     all_confidence_predictions.append({
#                         "qid": row.get("qid", f"sample_{batch_indices[i]}"),
#                         "confidence_letter": conf_pred_letter,
#                         "verbal_confidence": verbal_confidence,
#                         "confidence_probs": conf_probs[i].cpu().tolist(),
#                         "confidence_entropy": conf_entropy,
#                     })
                    
#                     # Log both MCQ and confidence in the same log file
#                     if log_file_path:
#                         # Log MCQ assessment
#                         mcq_log_entry = {
#                             "type": "mcq_accuracy_assessment",
#                             "validation_step": validation_step,
#                             "qid": row.get("qid", f"sample_{batch_indices[i]}"),
#                             "question": row.get("question", ""),
#                             "full_prompt": answer_prompts[i],  # Full prompt as model sees it
#                             "model_response_token_id": int(mcq_response_token_id),
#                             "model_response_logit": float(mcq_response_logit),  # Logit value for the predicted token
#                             "model_response_text": mcq_response_text,  # Exact decoded response from model (argmax of full vocab)
#                             "correct_answer_letter": correct_letter,
#                             "correct_answer_text": correct_answer_text,
#                             "model_answer_letter": predicted_letter,
#                             "model_answer_text": predicted_answer_text,
#                             "is_correct_by_text": bool(is_correct_by_text),
#                             "is_correct_by_letter": bool(is_correct_by_letter),
#                             "entropy": float(entropy),
#                             "probabilities": {
#                                 "A": float(answer_probs[i][0].item()),
#                                 "B": float(answer_probs[i][1].item()),
#                                 "C": float(answer_probs[i][2].item()),
#                                 "D": float(answer_probs[i][3].item()),
#                             },
#                             "all_options": {
#                                 "A": options.get("A", ""),
#                                 "B": options.get("B", ""),
#                                 "C": options.get("C", ""),
#                                 "D": options.get("D", ""),
#                             },
#                             "timestamp": datetime.now(timezone.utc).isoformat()
#                         }
#                         write_log(log_file_path, mcq_log_entry)
                        
#                         # Log confidence assessment
#                         conf_log_entry = {
#                             "type": "confidence_assessment",
#                             "validation_step": validation_step,
#                             "qid": row.get("qid", f"sample_{batch_indices[i]}"),
#                             "question": row.get("question", ""),
#                             "full_prompt": confidence_prompts[i],  # Full prompt as model sees it
#                             "model_response_token_id": int(conf_response_token_id),
#                             "model_response_logit": float(conf_response_logit),  # Logit value for the predicted token
#                             "model_response_text": conf_response_text,  # Exact decoded response from model (argmax of full vocab)
#                             "confidence_predicted_letter": conf_pred_letter,
#                             "verbal_confidence": float(verbal_confidence),
#                             "confidence_entropy": float(conf_entropy),
#                             "confidence_probabilities": {
#                                 "A": float(conf_probs[i][0].item()),
#                                 "B": float(conf_probs[i][1].item()),
#                                 "C": float(conf_probs[i][2].item()),
#                                 "D": float(conf_probs[i][3].item()),
#                                 "E": float(conf_probs[i][4].item()),
#                                 "F": float(conf_probs[i][5].item()),
#                                 "G": float(conf_probs[i][6].item()),
#                                 "H": float(conf_probs[i][7].item()),
#                             },
#                             "timestamp": datetime.now(timezone.utc).isoformat()
#                         }
#                         write_log(log_file_path, conf_log_entry)
                
#                 # Clear batch
#                 batch = []
#                 batch_indices = []
    
#     # Calculate aggregate metrics
#     accuracy = np.mean(all_correct) if all_correct else 0.0
#     avg_entropy = np.mean(all_entropies) if all_entropies else 0.0
#     std_entropy = np.std(all_entropies) if all_entropies else 0.0
    
#     # DIAGNOSTIC: Report average logit values to detect token bias
#     print(f"\n  DIAGNOSTIC: Average raw logits per token (across all batches):")
#     avg_logits = {}
#     for letter in "ABCD":
#         if logit_counts[letter] > 0:
#             avg_logits[letter] = logit_sums[letter] / logit_counts[letter]
#             print(f"    {letter} (token {mcq_tokens[letter]}): {avg_logits[letter]:.4f}")
    
#     # Check if there's a significant logit bias
#     if avg_logits:
#         logit_values = list(avg_logits.values())
#         max_logit_letter = max(avg_logits, key=avg_logits.get)
#         min_logit_letter = min(avg_logits, key=avg_logits.get)
#         logit_range = max(logit_values) - min(logit_values)
#         if logit_range > 1.0:  # More than 1.0 logit difference suggests bias
#             print(f"  ⚠️  WARNING: Significant logit bias detected (range={logit_range:.4f})")
#             print(f"     Highest: {max_logit_letter} ({avg_logits[max_logit_letter]:.4f}), "
#                   f"Lowest: {min_logit_letter} ({avg_logits[min_logit_letter]:.4f})")
#             print(f"     This may explain why the model favors certain answers.")
#         else:
#             print(f"  ✓ Logit values are relatively balanced (range={logit_range:.4f})")
    
#     # Calculate average logits for each token (diagnostic for bias detection)
#     avg_logits = {}
#     for letter in "ABCD":
#         if logit_counts[letter] > 0:
#             avg_logits[letter] = logit_sums[letter] / logit_counts[letter]
#         else:
#             avg_logits[letter] = 0.0
    
#     print(f"\n{'='*80}")
#     print(f"MCQ Accuracy Assessment Results:")
#     print(f"{'='*80}")
#     print(f"  Total questions: {len(all_correct)}")
#     print(f"  Correct answers: {sum(all_correct)}")
#     print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
#     print(f"  Expected random accuracy: 0.2500 (25.00%)")
#     print(f"  Average Entropy: {avg_entropy:.4f}")
#     print(f"\n  Average Logits (diagnostic for token bias):")
#     for letter in "ABCD":
#         print(f"    {letter}: {avg_logits[letter]:.4f}")
#     print(f"\n  Model's predicted MCQ answer distribution (A-D only, NOT confidence A-H):")
#     for letter in "ABCD":
#         count = predicted_letter_counts.get(letter, 0)
#         pct = (count / len(all_correct) * 100) if all_correct else 0
#         expected_pct = 25.0  # Expected if uniform
#         diff = pct - expected_pct
#         print(f"    {letter}: {count:4d} ({pct:5.2f}%) [expected ~25%, diff: {diff:+.2f}%]")
#     print(f"\n  Correct answer letter distribution (A-D only):")
#     print(f"    (After shuffling, should be ~25% each if shuffling works correctly)")
#     for letter in "ABCD":
#         count = correct_letter_counts.get(letter, 0)
#         pct = (count / len(all_correct) * 100) if all_correct else 0
#         expected_pct = 25.0  # Expected if shuffling is uniform
#         diff = pct - expected_pct
#         print(f"    {letter}: {count:4d} ({pct:5.2f}%) [expected ~25%, diff: {diff:+.2f}%]")
    
#     # DIAGNOSTIC: Check if shuffling is working (correct answers should be evenly distributed)
#     correct_dist_std = np.std([correct_letter_counts.get(letter, 0) / len(all_correct) * 100 
#                                 for letter in "ABCD"]) if all_correct else 0.0
#     if correct_dist_std > 5.0:  # More than 5% standard deviation suggests shuffling issue
#         print(f"\n  ⚠️  WARNING: Correct answer distribution has high variance (std={correct_dist_std:.2f}%)")
#         print(f"     This suggests shuffling may not be working correctly or dataset has bias.")
#     else:
#         print(f"\n  ✓ Correct answer distribution is uniform (std={correct_dist_std:.2f}%), shuffling appears to work.")
    
#     # DIAGNOSTIC: Check if model predictions are biased
#     pred_dist_std = np.std([predicted_letter_counts.get(letter, 0) / len(all_correct) * 100 
#                              for letter in "ABCD"]) if all_correct else 0.0
#     if pred_dist_std > 10.0:  # More than 10% standard deviation suggests model bias
#         print(f"  ⚠️  WARNING: Model predictions have high variance (std={pred_dist_std:.2f}%)")
#         print(f"     This suggests the model has learned a position bias or token bias.")
#     else:
#         print(f"  ✓ Model predictions are relatively uniform (std={pred_dist_std:.2f}%).")
    
#     print(f"{'='*80}\n")
    
#     # Log to W&B
#     try:
#         import wandb
#         wandb.log({
#             "val/accuracy": accuracy,
#             "val/answer_entropy": avg_entropy,
#         }, step=validation_step)
#     except (ImportError, AttributeError):
#         pass  # Silently fail if wandb not available
    
#     return {
#         "accuracy": accuracy,
#         "avg_entropy": avg_entropy,
#         "std_entropy": std_entropy,
#         "predicted_letter_counts": predicted_letter_counts,
#         "correct_letter_counts": correct_letter_counts,
#         "total_questions": len(all_correct),
#     }

