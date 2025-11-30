import json
import os
import re
import random
import numpy as np
import traceback
from datetime import datetime, timezone
from torch.utils.data import Dataset



seed_value = 42
rng = np.random.default_rng(seed_value)

def get_batch(dataset, batch_size):
    idxs = rng.choice(len(dataset), size=batch_size, replace=False)
    return [dataset[i] for i in idxs]


def write_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")

        
def load_jsonl_dataset(path):
    """
    Load the PopMC dataset and convert each entry into the canonical MCQ format:
        - options: A, B, C, D
        - correct_letter: 'A' … 'D'
    """
    out = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)

            question = row["question"]
            correct = row["correct_answer"]
            distractors = row["distractors"]

            # Build 4-option MCQ set
            opts = [correct] + distractors
            if len(opts) != 4:
                # Skip malformed items
                continue

            # Shuffle + keep track of correct letter
            letters = ["A", "B", "C", "D"]
            paired = list(zip(letters, opts))
            random.shuffle(paired)

            shuffled_letters, shuffled_opts = zip(*paired)
            options = {L: O for L, O in paired}

            # Find which letter contains the correct answer
            for L, O in options.items():
                if O == correct:
                    correct_letter = L
                    break

            out.append({
                "qid": row.get("qid"),
                "question": question,
                "options": options,
                "correct_letter": correct_letter,
            })

    return out



class MCQDataset(Dataset):
    """Dataset for Multiple Choice Questions."""

    def __init__(self, path):
        """
        Initialize MCQ dataset from JSONL file.

        Args:
            path: Path to JSONL file containing questions
        """
        with open(path, 'r', encoding='utf-8') as f:
            self.rows = [json.loads(line) for line in f]

    def __len__(self):
        """Return number of questions in dataset."""
        return len(self.rows)

    def __getitem__(self, idx):
        """
        Get a single question from the dataset.

        Args:
            idx: Index of question to retrieve

        Returns:
            Dictionary with question, options, and correct answer
        """
        row = self.rows[idx]

        question = row["question"]
        correct = row["correct_answer"]
        distractors = row["distractors"][:3]  # ensure 3

        # No shuffling: fixed layout
        options = [(correct, True)] + [(d, False) for d in distractors]

        labeled = {}
        correct_letter = None
        for label, (text, is_correct) in zip("ABCD", options):
            labeled[label] = text
            if is_correct:
                correct_letter = label

        return {
            "qid": row.get("qid", str(idx)),
            "question": question,
            "options": labeled,
            "correct_letter": correct_letter,
        }


def normalize_text(s):
    """Normalize text for comparison: lowercase, strip, normalize whitespace, remove trailing period."""
    if not s:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".")
    return s


def load_mcq_results_data(mcq_results_path, log_file_path=None):
    """
    Load MCQ results data from JSON or JSONL file.
    Returns a dictionary mapping question ID to result data, plus a text-to-ID mapping.
    
    Structure:
    - results_lookup: {qid: result_data} - primary storage by ID
    - text_to_id: {normalized_text: qid} - mapping from text to ID for fallback lookup
    """
    if mcq_results_path is None:
        print("mcq_results_path is None")
        return None
    
    results_lookup = {}  # Primary storage: {qid: result_data}
    text_to_id = {}  # Secondary mapping: {normalized_text: qid} for text-based lookup
    unique_question_count = 0  # Track unique questions
    
    try:
        # Try different encodings in case of BOM or encoding issues
        encodings = ['utf-8', 'utf-8-sig', 'latin-1']
        file_handle = None
        
        for encoding in encodings:
            try:
                file_handle = open(mcq_results_path, 'r', encoding=encoding)
                # Read first few bytes to check format
                first_bytes = file_handle.read(1024)
                file_handle.seek(0)
                
                # Strip BOM if present
                if first_bytes.startswith('\ufeff'):
                    first_bytes = first_bytes[1:]
                    file_handle.seek(1)
                
                first_char = first_bytes.strip()[0] if first_bytes.strip() else ''
                break
            except Exception as e:
                if file_handle:
                    file_handle.close()
                continue
        
        if file_handle is None:
            raise IOError(f"Could not open file with any encoding: {encodings}")
        
        try:
            if first_char == '{':
                # Likely JSON format - try to parse as single JSON object
                print("Attempting to parse as JSON format...")
                try:
                    data = json.load(file_handle)
                    if isinstance(data, dict) and "results" in data:
                        # JSON format with results dictionary
                        print(f"Found 'results' dictionary with {len(data['results'])} entries")
                        for qid, result in data["results"].items():
                            unique_question_count += 1
                            question_data = result.get("question", {})
                            question_id = question_data.get("id", qid)
                            question_text = question_data.get("question", "")
                            
                            # Extract options from question_data if available
                            options = question_data.get("options", {})
                            
                            # Store once by ID (preferred) or by text if no ID
                            result_data = {
                                "subject_answer": result.get("subject_answer"),
                                "probs": result.get("probs", {}),
                                "options": options
                            }
                            
                            if question_id:
                                results_lookup[str(question_id)] = result_data
                                # Also create text-to-ID mapping for fallback lookup
                                if question_text:
                                    norm_text = normalize_text(question_text)
                                    if norm_text:
                                        text_to_id[norm_text] = str(question_id)
                            elif question_text:
                                # No ID available, use normalized text as key
                                norm_text = normalize_text(question_text)
                                if norm_text:
                                    results_lookup[norm_text] = result_data
                    else:
                        # Single result object, not wrapped in "results"
                        unique_question_count += 1
                        question_data = data.get("question", {})
                        question_id = question_data.get("id")
                        question_text = question_data.get("question", "")
                        
                        result_data = {
                            "subject_answer": data.get("subject_answer"),
                            "probs": data.get("probs", {})
                        }
                        
                        if question_id:
                            results_lookup[str(question_id)] = result_data
                            if question_text:
                                norm_text = normalize_text(question_text)
                                if norm_text:
                                    text_to_id[norm_text] = str(question_id)
                        elif question_text:
                            norm_text = normalize_text(question_text)
                            if norm_text:
                                results_lookup[norm_text] = result_data
                except json.JSONDecodeError as json_err:
                    print(f"JSON parsing error: {json_err}")
                    error_pos = json_err.pos if hasattr(json_err, 'pos') else None
                    print(f"Error at position {error_pos}")
                    
                    # Try to parse the results dictionary manually by extracting entries
                    print("Attempting to extract results entries manually...")
                    file_handle.seek(0)
                    content = file_handle.read()
                    
                    # Try to find and parse the "results" dictionary
                    # Look for the pattern: "results": {
                    results_start = content.find('"results": {')
                    if results_start != -1:
                        # Find the opening brace of results
                        brace_start = content.find(
                            '{', results_start + len('"results": '))
                        if brace_start != -1:
                            # Try to extract individual entries
                            # Pattern: "qid": { ... }
                            # Match pattern: "key": { ... } where key is a question ID
                            # This regex matches a quoted key followed by a JSON object
                            pattern = r'"([^"]+)":\s*\{'
                            matches = list(re.finditer(pattern, content[brace_start:]))
                            
                            print(f"Found {len(matches)} potential result entries")
                            
                            # Try to extract each entry
                            for i, match in enumerate(matches):
                                entry_start = brace_start + match.end() - 1  # -1 to include the {
                                # Find the matching closing brace
                                brace_count = 0
                                entry_end = entry_start
                                for j, char in enumerate(content[entry_start:], start=entry_start):
                                    if char == '{':
                                        brace_count += 1
                                    elif char == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            entry_end = j + 1
                                            break
                                
                                if entry_end > entry_start:
                                    try:
                                        entry_json = content[entry_start:entry_end]
                                        result = json.loads(entry_json)
                                        
                                        # Ensure result is a dictionary
                                        if not isinstance(result, dict):
                                            continue
                                        
                                        unique_question_count += 1
                                        question_data = result.get("question", {})
                                        
                                        # Ensure question_data is a dictionary
                                        if not isinstance(question_data, dict):
                                            # If question_data is not a dict, try to use the match group as question_id
                                            question_id = match.group(1)
                                            question_text = ""
                                        else:
                                            question_id = question_data.get("id", match.group(1))
                                            question_text = question_data.get("question", "")
                                        
                                        result_data = {
                                            "subject_answer": result.get("subject_answer"),
                                            "probs": result.get("probs", {})
                                        }
                                        
                                        if question_id:
                                            results_lookup[str(question_id)] = result_data
                                            if question_text:
                                                norm_text = normalize_text(question_text)
                                                if norm_text:
                                                    text_to_id[norm_text] = str(question_id)
                                        elif question_text:
                                            norm_text = normalize_text(question_text)
                                            if norm_text:
                                                results_lookup[norm_text] = result_data
                                    except (json.JSONDecodeError, AttributeError, TypeError) as e:
                                        # Skip this entry if it can't be parsed or has wrong structure
                                        continue
                                
                                # Progress update
                                if (i + 1) % 1000 == 0:
                                    print(f"Processed {i + 1} entries, found {len(results_lookup)} valid results so far...")
                            
                            if len(results_lookup) > 0:
                                print(f"Successfully extracted {len(results_lookup)} results from malformed JSON")
                            else:
                                print("Could not extract any results from malformed JSON")
                                # Fall back to JSONL attempt
                                print("Attempting to parse as JSONL format...")
                                file_handle.seek(0)
                                line_num = 0
                                for line in file_handle:
                                    line_num += 1
                                    if line.strip():
                                        try:
                                            result = json.loads(line)
                                            unique_question_count += 1
                                            question_data = result.get("question", {})
                                            question_id = question_data.get("id")
                                            question_text = question_data.get("question", "")
                                            
                                            result_data = {
                                                "subject_answer": result.get("subject_answer"),
                                                "probs": result.get("probs", {})
                                            }
                                            
                                            if question_id:
                                                results_lookup[str(question_id)] = result_data
                                                if question_text:
                                                    norm_text = normalize_text(question_text)
                                                    if norm_text:
                                                        text_to_id[norm_text] = str(question_id)
                                            elif question_text:
                                                norm_text = normalize_text(question_text)
                                                if norm_text:
                                                    results_lookup[norm_text] = result_data
                                        except json.JSONDecodeError as line_err:
                                            if line_num <= 5:  # Only print first few errors
                                                print(f"Warning: Failed to parse line {line_num}: {line_err}")
                        else:
                            print("Could not find results dictionary in file")
                    else:
                        print("Could not find 'results' key in file")
                        # Fall back to JSONL attempt
                        print("Attempting to parse as JSONL format...")
                        file_handle.seek(0)
                        line_num = 0
                        for line in file_handle:
                            line_num += 1
                            if line.strip():
                                try:
                                    result = json.loads(line)
                                    unique_question_count += 1
                                    question_data = result.get("question", {})
                                    question_id = question_data.get("id")
                                    question_text = question_data.get("question", "")
                                    
                                    result_data = {
                                        "subject_answer": result.get("subject_answer"),
                                        "probs": result.get("probs", {})
                                    }
                                    
                                    if question_id:
                                        results_lookup[str(question_id)] = result_data
                                        if question_text:
                                            norm_text = normalize_text(question_text)
                                            if norm_text:
                                                text_to_id[norm_text] = str(question_id)
                                    elif question_text:
                                        norm_text = normalize_text(question_text)
                                        if norm_text:
                                            results_lookup[norm_text] = result_data
                                except json.JSONDecodeError as line_err:
                                    if line_num <= 5:  # Only print first few errors
                                        print(f"Warning: Failed to parse line {line_num}: {line_err}")
            else:
                # Likely JSONL format - one JSON object per line
                print("Attempting to parse as JSONL format...")
                line_num = 0
                for line in file_handle:
                    line_num += 1
                    if line.strip():
                        try:
                            result = json.loads(line)
                            unique_question_count += 1
                            question_data = result.get("question", {})
                            question_id = question_data.get("id")
                            question_text = question_data.get("question", "")
                            
                            # Extract options from question_data if available
                            options = question_data.get("options", {})
                            
                            result_data = {
                                "subject_answer": result.get("subject_answer"),
                                "probs": result.get("probs", {}),
                                "options": options
                            }
                            
                            if question_id:
                                results_lookup[str(question_id)] = result_data
                                if question_text:
                                    norm_text = normalize_text(question_text)
                                    if norm_text:
                                        text_to_id[norm_text] = str(question_id)
                            elif question_text:
                                norm_text = normalize_text(question_text)
                                if norm_text:
                                    results_lookup[norm_text] = result_data
                        except json.JSONDecodeError as line_err:
                            if line_num <= 5:  # Only print first few errors
                                print(f"Warning: Failed to parse line {line_num}: {line_err}")
        finally:
            file_handle.close()
            
    except Exception as e:
        print(f"Warning: Failed to load MCQ results data from "
              f"{mcq_results_path}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None
    
    # Attach text-to-ID mapping to the lookup dict for use in verify_and_resolve_options
    # We'll use a special key that won't conflict with question IDs
    results_lookup["__text_to_id__"] = text_to_id
    
    # Report loading statistics
    if unique_question_count > 0:
        print(f"Loaded {unique_question_count} unique questions")
        print(f"  Created {len(results_lookup) - 1} primary lookup entries (stored by ID, with text-to-ID mapping for fallback)")
    else:
        print(f"Loaded {len(results_lookup) - 1 if '__text_to_id__' in results_lookup else len(results_lookup)} MCQ lookup entries")
    
    if log_file_path:
        # Import here to avoid circular import
        from finetune_utils import write_log
        write_log(log_file_path, {
            "message": f"Loaded {unique_question_count} unique questions",
            "lookup_entries": len(results_lookup) - 1 if '__text_to_id__' in results_lookup else len(results_lookup),
            "text_to_id_mappings": len(text_to_id),
            "note": "Questions stored once by ID, with text-to-ID mapping for flexible lookup"
        })

    return results_lookup


def verify_and_resolve_options(row, mcq_results_lookup, log_file_path=None):
    """
    Resolve the correct option set for the batch question.

    Checks top-level keys created by the loader above.

    Args:
        row: Dictionary with question data
        mcq_results_lookup: Lookup dictionary for recorded results
        log_file_path: Path for logging (optional)

    Returns:
        Tuple of (result_data, options)
    """
    qid = row.get("qid")
    batch_question = row["question"]
    # These are the raw/shuffled ones from dataset
    batch_opts = row["options"]

    # Default: use batch opts
    if not mcq_results_lookup:
        return None, batch_opts

    # ---------- 1. Lookup ----------
    result_data = None
    
    # Extract text-to-ID mapping if available
    text_to_id = mcq_results_lookup.get("__text_to_id__", {})
    
    # Try QID match first
    if qid is not None and str(qid) in mcq_results_lookup:
        result_data = mcq_results_lookup[str(qid)]
    
    # Try Text match if QID failed - use text-to-ID mapping to find the ID, then look up by ID
    if result_data is None:
        norm_text = normalize_text(batch_question)
        if norm_text in text_to_id:
            # Found text mapping, look up by the mapped ID
            mapped_id = text_to_id[norm_text]
            if mapped_id in mcq_results_lookup:
                result_data = mcq_results_lookup[mapped_id]
        elif norm_text in mcq_results_lookup:
            # Fallback: text might be stored directly (for questions without IDs)
            result_data = mcq_results_lookup[norm_text]

    # Nothing found → fallback
    if result_data is None:
        if log_file_path:
            # Import here to avoid circular import
            from finetune_utils import write_log
            write_log(log_file_path, {
                "type": "verification_no_lookup_found",
                "qid": qid,
                "question_snippet": batch_question[:50],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        return None, batch_opts

    # ---------- 2. Extract Fields (Robustly) ----------
    # The loader puts these at the top level now
    rec_opts = result_data.get("options")
    rec_q_text = result_data.get("question_text")

    # If missing options, we can't proceed
    if not rec_opts:
        if log_file_path:
            # Import here to avoid circular import
            from finetune_utils import write_log
            write_log(log_file_path, {
                "type": "verification_missing_options_in_record",
                "qid": qid,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        return None, batch_opts

    # ---------- 3. Verify Question Text ----------
    # Only verify if we successfully captured text in the loader
    if rec_q_text:
        if normalize_text(batch_question) != normalize_text(rec_q_text):
            if log_file_path:
                # Import here to avoid circular import
                from finetune_utils import write_log
                write_log(log_file_path, {
                    "type": "verification_question_text_mismatch",
                    "qid": qid,
                    "batch": batch_question,
                    "recorded": rec_q_text,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            return None, batch_opts

    # ---------- 4. Verify/Sync Options ----------
    # We rely on the recorded options (rec_opts) as the truth.
    # We just ensure the batch *contains* the same keys so we don't crash.
    for letter in ["A", "B", "C", "D"]:
        if letter not in rec_opts:
            # If the recorded data is missing a letter (rare, but possible in broken data)
            return None, batch_opts

        # Optional: Verify content matches if you suspect the dataset changed
        # This might fail if "options" in batch are shuffled vs "options" in record
        # Since we want to FORCE the record, we often skip checking the content 
        # unless we are unsure if it's the same question.
        # Given we checked QID and Text above, we trust rec_opts.

    # Verification passed - log success
    if log_file_path:
        # Import here to avoid circular import
        from finetune_utils import write_log
        write_log(log_file_path, {
            "type": "verification_passed",
            "qid": qid,
            "question_snippet": batch_question[:50],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    return result_data, rec_opts


def validate_file_exists_and_not_empty(file_path, file_description="file"):
    """
    Validate that a file exists and is not empty.
    
    Args:
        file_path: Path to the file to validate
        file_description: Description of the file for error messages
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_description.capitalize()} not found: {file_path}\n"
            f"Please provide a valid path to your {file_description}."
        )
    
    # Check if file is empty
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError(
            f"{file_description.capitalize()} is empty: {file_path}\n"
            f"The file exists but contains no data. Please check your {file_description} file."
        )


def validate_and_load_dataset(data_path, dataset_type="training"):
    """
    Validate and load a dataset file.
    
    Args:
        data_path: Path to JSONL dataset file
        dataset_type: Type of dataset ("training" or "validation")
        
    Returns:
        MCQDataset instance
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If dataset is empty
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{dataset_type.capitalize()} data file not found: {data_path}\n"
            f"Please provide a valid path to your {dataset_type} dataset."
        )
    
    dataset = MCQDataset(data_path)
    if len(dataset) == 0:
        raise ValueError(
            f"{dataset_type.capitalize()} dataset is empty: {data_path}\n"
            f"The file exists but contains no data. Please check your dataset file."
        )
    
    return dataset


def resolve_file_path(file_path, search_dirs=None):
    """
    Try to find a file in common locations if it doesn't exist at the given path.
    
    Args:
        file_path: Original file path
        search_dirs: List of additional directories to search (optional)
        
    Returns:
        Resolved file path if found, original path otherwise
        
    Raises:
        FileNotFoundError: If file cannot be found in any location
    """
    if os.path.exists(file_path):
        return file_path
    
    file_path = file_path.strip()
    basename = os.path.basename(file_path)
    
    # Default search directories
    default_search_dirs = [
        "data",
        "explicit_confidence_task_logs",
        "capabilities_test_logs",
    ]
    
    if search_dirs:
        search_dirs = list(search_dirs) + default_search_dirs
    else:
        search_dirs = default_search_dirs
    
    # Build possible paths
    possible_paths = [file_path]
    for search_dir in search_dirs:
        possible_paths.append(os.path.join(search_dir, basename))
    possible_paths.append(os.path.join(os.getcwd(), file_path))
    
    # Try each path
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If not found, raise error with helpful message
    error_msg = (
        f"Error: Could not find file. "
        f"Tried the following paths:\n"
        + "\n".join(f"  - {path}" for path in possible_paths)
        + f"\n\nOriginal path: {file_path}"
        + f"\nCurrent working directory: {os.getcwd()}"
    )
    raise FileNotFoundError(error_msg)


def validate_training_files(args):
    """
    Validate all input files for training, with automatic path resolution.
    
    This function validates:
    - Training data file (required)
    - Validation data file (optional)
    - Test data file (optional, with path resolution)
    - MCQ results file (optional, with path resolution)
    
    Args:
        args: Argument object with the following attributes:
            - train_data_path: Path to training data (required)
            - val_data_path: Path to validation data (optional)
            - test_data_path: Path to test data (optional)
            - mcq_results_data: Path to MCQ results (optional)
            
    Modifies args in place by resolving paths for test_data_path and mcq_results_data.
    
    Raises:
        FileNotFoundError: If any required file is not found
        ValueError: If any file is empty
    """
    print("\n=== Validating input files ===")
    
    # Validate training data file (required)
    validate_file_exists_and_not_empty(args.train_data_path, "training data file")
    print(f"✓ Training data file exists and is not empty: {args.train_data_path}")
    
    # Validate validation data file (optional)
    if args.val_data_path:
        validate_file_exists_and_not_empty(args.val_data_path, "validation data file")
        print(f"✓ Validation data file exists and is not empty: {args.val_data_path}")
    
    # Validate test data file (optional, with path resolution)
    if args.test_data_path:
        try:
            args.test_data_path = resolve_file_path(
                args.test_data_path,
                search_dirs=["data"]  # Test files are typically in data/
            )
        except FileNotFoundError as e:
            print(f"Test data file not found at: {args.test_data_path}")
            print(f"Current working directory: {os.getcwd()}")
            print("Attempting to find file...")
            raise FileNotFoundError(
                f"Test data file not found: {args.test_data_path}\n"
                f"Please provide a valid path to the test data file."
            ) from e
        
        validate_file_exists_and_not_empty(args.test_data_path, "test data file")
        print(f"✓ Test data file exists and is not empty: {args.test_data_path}")
    
    # Validate MCQ results file (optional, with path resolution)
    if args.mcq_results_data:
        try:
            args.mcq_results_data = resolve_file_path(
                args.mcq_results_data,
                search_dirs=["data", "explicit_confidence_task_logs", "capabilities_test_logs"]
            )
        except FileNotFoundError as e:
            print(f"MCQ results file not found at: {args.mcq_results_data}")
            print(f"Current working directory: {os.getcwd()}")
            print("Attempting to find file...")
            raise FileNotFoundError(
                f"MCQ results file not found: {args.mcq_results_data}\n"
                f"Please provide a valid path to the MCQ results file."
            ) from e
        
        validate_file_exists_and_not_empty(args.mcq_results_data, "MCQ results file")
        print(f"✓ MCQ results file exists and is not empty: {args.mcq_results_data}")
    
    print("=== File validation complete ===\n")


def collate_fn(batch):
    """Custom collate function that returns a list of dictionaries."""
    return batch

