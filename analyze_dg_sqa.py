import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
import json
import os
import numpy as np
from load_and_format_datasets import load_and_format_dataset
import re
from collections import defaultdict
from logres_helpers import *
from pathlib import Path

res_dicts = defaultdict(dict)

FIRST_PASS = True
def log_output(message_string, print_to_console=False, suppress=True):
    global FIRST_PASS
    if FIRST_PASS:
        openstr = "w"
        FIRST_PASS = False
    else:
        openstr = "a"
    if print_to_console:
        print(message_string)
    if suppress: return
    with open(LOG_FILENAME, openstr, encoding='utf-8') as f:
        f.write(str(message_string) + "\n")

def prepare_regression_data_for_model(game_file_paths_list,
                                      sqa_feature_lookup,
                                      capabilities_s_i_map_for_model,
                                      a_i_map_for_this_model,
                                      p_i_map_for_this_model=None,
                                      entropy_map_for_this_model=None,
                                      o_map_for_this_model=None,
                                      sp_map_for_this_model=None,
                                      game_file_suffix=""):
    all_regression_data_for_model = []
    file_level_features_cache = []

    if not game_file_paths_list:
        return None

    phase2_corcnt, phase2_totalcnt = 0, 0
    for game_file_path in game_file_paths_list:
        judgment_data, teammate_judgment_data = {}, {}
        try:
            with open(game_file_path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)

            judgment_file_path = game_file_path.replace(f"_game_data{game_file_suffix}.json", f"_game_data{game_file_suffix}_judgment_judge_data.json")
            if os.path.exists(judgment_file_path):
                try:
                    with open(judgment_file_path, 'r', encoding='utf-8') as jf:
                        judgment_content = json.load(jf)
                    if isinstance(judgment_content, dict) and "results" in judgment_content and isinstance(judgment_content["results"], dict):
                        for qid, q_data in judgment_content["results"].items():
                            if isinstance(q_data, dict) and "delegate" in q_data:
                                judgment_data[qid] = q_data["delegate"]
                except Exception as e_judge:
                    print(f"Error loading or parsing judgment file {judgment_file_path}: {e_judge}")
            judgment_file_path = game_file_path.replace(f"_game_data{game_file_suffix}.json", "_game_data{game_file_suffix}_teammatejudgment_judge_data.json")
            if os.path.exists(judgment_file_path):
                try:
                    with open(judgment_file_path, 'r', encoding='utf-8') as jf:
                        judgment_content = json.load(jf)
                    if isinstance(judgment_content, dict) and "results" in judgment_content and isinstance(judgment_content["results"], dict):
                        for qid, q_data in judgment_content["results"].items():
                            if isinstance(q_data, dict) and "delegate" in q_data:
                                teammate_judgment_data[qid] = q_data["delegate"]
                except Exception as e_judge:
                    print(f"Error loading or parsing teammate judgment file {judgment_file_path}: {e_judge}")

        except Exception as e:
            print(f"Error loading game file {game_file_path}: {e}")
            continue

        filename_base = os.path.basename(game_file_path)
        phase2_trials = [t for t in game_data.get("results", []) if t.get('phase',0) == 2 or (t.get('passes_used') is not None and t['passes_used'] >= 0)]

        phase2_corcnt += sum(1 for t in phase2_trials if t["delegation_choice"]=="Self" and t["subject_correct"])
        phase2_totalcnt += sum(1 for t in phase2_trials if t["delegation_choice"]=="Self")

        if phase2_trials:
            file_level_features_cache.append({
                "trials": phase2_trials,
                "teammate_accuracy_phase1_file": game_data.get("teammate_accuracy_phase1",None),
                "subject_accuracy_phase1_file": game_data.get("subject_accuracy_phase1",None),
                "summary_file": "_summary_" in filename_base,
                "nobio_file": "_nobio_" in filename_base,
                "noeasy_file": "_noeasy_" in filename_base,
                "noctr_file": "_noctr_" in filename_base,
                "judgment_data": judgment_data,
                "teammate_judgment_data": teammate_judgment_data
            })

    if not file_level_features_cache:
        print(f"No valid game data found in the provided files.")
        return None

    subject_acc_for_ratio_calc = next((f["subject_accuracy_phase1_file"] 
                                       for f in file_level_features_cache 
                                       if f["subject_accuracy_phase1_file"] is not None), None)

    all_teammate_accs = [f["teammate_accuracy_phase1_file"] for f in file_level_features_cache if f["teammate_accuracy_phase1_file"] is not None]
    create_teammate_skill_ratio_reg = len(set(all_teammate_accs)) > 1 and \
                                     subject_acc_for_ratio_calc is not None and \
                                     subject_acc_for_ratio_calc != 0
    
    create_summary_reg = len(set(f["summary_file"] for f in file_level_features_cache)) > 1
    create_nobio_reg = len(set(f["nobio_file"] for f in file_level_features_cache)) > 1
    create_noeasy_reg = len(set(f["noeasy_file"] for f in file_level_features_cache)) > 1
    create_noctr_reg = len(set(f["noctr_file"] for f in file_level_features_cache)) > 1

    teammate_accs_phase1 = []
    for file_ctr, file_data in enumerate(file_level_features_cache):
        print(f"\nProcessing file {file_ctr + 1}/{len(file_level_features_cache)}: {game_file_paths_list[file_ctr]}")
        print(f"len(file_data['trials']) = {len(file_data['trials'])}")
        if  file_data["teammate_accuracy_phase1_file"] is not None:
            teammate_accs_phase1.append(file_data["teammate_accuracy_phase1_file"])

        for trial in file_data["trials"]:
            q_id = trial.get("question_id")
            delegation_choice_str = trial.get("delegation_choice")
            if delegation_choice_str == "Self":
                subject_answer = trial.get("subject_answer")
                if subject_answer and re.search(r'\b(T|DELEGATE)\b', subject_answer, re.IGNORECASE) and \
                   trial.get("evaluation_method", "").startswith("llm_plurality") and "not delegate" not in subject_answer.lower() and "rather than delegate" not in subject_answer.lower():
                    judg_dict = trial.get("judgments")
                    if judg_dict and sum(j == "NOT ATTEMPTED" for j in judg_dict.values()) > len(judg_dict) / 2:
                        delegation_choice_str = "Teammate"
                        phase2_totalcnt -= 1
            
            if not q_id or not delegation_choice_str:
                print(f"Skipping trial with missing q_id or delegation_choice: {trial}")
                continue

            prob_dict_trial = trial.get("probs")
            max_norm_prob_trial = None
            norm_prob_entropy_trial = None
            t_prob = None
            if isinstance(prob_dict_trial, dict):
                if '_decisonOnly' in game_file_paths_list[0]:
                    digitmap_dict_trial = trial.get("digit_mapping")
                    loi = "1" if digitmap_dict_trial and digitmap_dict_trial.get("2") == "Answer" else "2"
                else:
                    loi = "T" if "delegate" in game_file_paths_list[0] else "P"
                non_t_probs_values = [float(v) for k, v in prob_dict_trial.items() if k != loi and isinstance(v, (int, float))]
                t_prob = prob_dict_trial[loi] if loi in prob_dict_trial and isinstance(prob_dict_trial[loi], (int, float)) else 0.0

                if non_t_probs_values:
                    sum_non_t_probs = sum(non_t_probs_values)
                    if sum_non_t_probs > 1e-9:
                        normalized_probs = [p / sum_non_t_probs for p in non_t_probs_values]
                        if normalized_probs:
                            max_norm_prob_trial = max(normalized_probs)
                            norm_prob_entropy_trial = -np.sum([p_norm * np.log2(p_norm) for p_norm in normalized_probs if p_norm > 1e-9])

            sqa_features = sqa_feature_lookup.get(q_id)
            s_i_capability = capabilities_s_i_map_for_model.get(q_id)
            p_i_capability = p_i_map_for_this_model.get(q_id) if p_i_map_for_this_model else None
            capabilities_entropy = entropy_map_for_this_model.get(q_id) if entropy_map_for_this_model else None

            if sqa_features and s_i_capability is not None:
                delegate_choice_numeric = 0 if delegation_choice_str == "Self" else 1
                subject_answer = trial.get("subject_answer")
                try:
                    subject_answer_full = trial['options'][subject_answer] if 'options' in trial and subject_answer and trial['options'] and isinstance(trial['options'], dict) and subject_answer in trial['options'] else subject_answer
                except Exception as e:
                    print(f"Error on qid {q_id} in file {game_file_paths_list[file_ctr]}, subject_answer={subject_answer}: {e}")
                    raise e                
                trial_data_dict = {
                    'q_id': q_id, 
                    'delegate_choice': delegate_choice_numeric,
                    's_i_capability': s_i_capability,
                    'subject_correct': False if trial.get('subject_correct') is None else trial['subject_correct'],
                    'answer_changed': False if a_i_map_for_this_model[q_id][1] == subject_answer_full else True,
                    'answer_type': sqa_features['answer_type'],
                    'q_length': np.log(len(sqa_features.get('q_text', '')) + 1e-9), # Add epsilon for empty q_text
                    'topic': sqa_features.get('topic', ''),
                    'judge_delegate': file_data.get("judgment_data", {}).get(q_id, np.nan),
                    'teammate_judge_delegate': file_data.get("teammate_judgment_data", {}).get(q_id, np.nan),
                    'p_i_capability': p_i_capability,
                    'o_prob': o_map_for_this_model.get(q_id) if o_map_for_this_model else None,
                    'sp_prob': sp_map_for_this_model.get(q_id) if sp_map_for_this_model else None,
                    'capabilities_entropy': capabilities_entropy,
                    "experiment_id": file_ctr,
                }

                trial_data_dict['change'] = 1 if trial_data_dict['delegate_choice'] == 0 and trial_data_dict['subject_correct'] != bool(trial_data_dict['s_i_capability']) else 0
                trial_data_dict['bad_change'] = 1 if trial_data_dict['delegate_choice'] == 0 and trial_data_dict['subject_correct'] == False and trial_data_dict['s_i_capability']==1 else 0

                if trial.get('team_correct') is not None:
                    trial_data_dict['team_correct'] = trial['team_correct']
                else: #set it randomly based on teammate_accuracy_phase1 (can happen when overruling delegate label based on not attempted)
                    if file_data["teammate_accuracy_phase1_file"] is not None:
                        trial_data_dict['team_correct'] = bool(np.random.binomial(1, file_data["teammate_accuracy_phase1_file"]))

                if max_norm_prob_trial is not None:
                    trial_data_dict['max_normalized_prob'] = max_norm_prob_trial
                if norm_prob_entropy_trial is not None:
                    trial_data_dict['normalized_prob_entropy'] = norm_prob_entropy_trial
                if t_prob is not None:
                    trial_data_dict['t_prob'] = t_prob

                if not np.isnan(trial_data_dict['judge_delegate']) and not np.isnan(trial_data_dict['teammate_judge_delegate']):
                    trial_data_dict['judge_delegate_combined'] = 0 if trial_data_dict['judge_delegate'] == 0 and trial_data_dict['teammate_judge_delegate'] == 1 else 1

                if create_teammate_skill_ratio_reg and file_data["teammate_accuracy_phase1_file"] is not None:
                    trial_data_dict['teammate_skill_ratio'] = file_data["teammate_accuracy_phase1_file"] / subject_acc_for_ratio_calc
                if file_data["teammate_accuracy_phase1_file"] is not None:
                    trial_data_dict['teammate_skill'] = file_data["teammate_accuracy_phase1_file"]
                
                if create_summary_reg: trial_data_dict['summary'] = int(file_data["summary_file"])
                if create_nobio_reg: trial_data_dict['nobio'] = int(file_data["nobio_file"])
                if create_noeasy_reg: trial_data_dict['noeasy'] = int(file_data["noeasy_file"])
                if create_noctr_reg: trial_data_dict['noctr'] = int(file_data["noctr_file"])
                
                all_regression_data_for_model.append(trial_data_dict)
            else:
                if not sqa_features:
                    print(f"Warning: No SQA features found for q_id {q_id} in file {game_file_paths_list[file_ctr]}. Skipping trial.")
                if s_i_capability is None:
                    print(f"Warning: No S_i capability found for q_id {q_id} in file {game_file_paths_list[file_ctr]}. Skipping trial.")
    if not all_regression_data_for_model:
        print(f"No valid regression data found in the provided game files.")
        return None
    
    df_to_return = pd.DataFrame(all_regression_data_for_model)

    if 'judge_delegate' in df_to_return.columns and not df_to_return['judge_delegate'].notna().any():
        df_to_return = df_to_return.drop(columns=['judge_delegate'])
    if 'teammate_judge_delegate' in df_to_return.columns and not df_to_return['teammate_judge_delegate'].notna().any():
        df_to_return = df_to_return.drop(columns=['teammate_judge_delegate'])

    print(f"\nPrepared regression DataFrame with {len(df_to_return)} rows and columns: {list(df_to_return.columns)}")    
    return df_to_return, subject_acc_for_ratio_calc, phase2_corcnt, phase2_totalcnt, np.mean(teammate_accs_phase1) if teammate_accs_phase1 else None

# --- File Grouping Logic ---
def get_feedback_status_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("feedback_config", {}).get("phase1_subject_feedback") is True
    except Exception:
        return False # Default on error

def split_by_feedback(file_list):
    true_files, false_files = [], []
    for f_path in file_list:
        (true_files if get_feedback_status_from_file(f_path) else false_files).append(f_path)
    result = []
    if true_files: result.append(("Feedback_True", true_files))
    if false_files: result.append(("Feedback_False", false_files)) # Catches False and None
    return result

def split_by_filename_attr(file_list, attr_check_func, name_if_true, name_if_false):
    true_files, false_files = [], []
    for f_path in file_list:
        (true_files if attr_check_func(os.path.basename(f_path)) else false_files).append(f_path)
    result = []
    if true_files: result.append((name_if_true, true_files))
    if false_files: result.append((name_if_false, false_files))
    return result

def process_file_groups(files_to_process, criteria_chain, model_name_for_log, group_path_names=None):
    group_path_names = group_path_names or ()

    if not criteria_chain:
        if files_to_process:
            yield group_path_names, files_to_process
        return

    current_criterion = criteria_chain[0]
    remaining_criteria = criteria_chain[1:]
    
    current_criterion_split_groups = current_criterion['split_logic'](files_to_process)

    if not current_criterion_split_groups and files_to_process:
        path_str = ", ".join([model_name_for_log] + list(group_path_names))
        parent_group_name = group_path_names[-1] if group_path_names else f"model {model_name_for_log}"
        indent = "  " * (len(group_path_names) + 2)
        print(f"{indent}No game files to process for {path_str} "
              f"after attempting to split by {current_criterion['name_prefix']}. "
              f"Skipping this {parent_group_name} group.")
        return

    for group_name, files_in_group in current_criterion_split_groups:
        indent_level = len(group_path_names) + 1
        print_indent = "  " * indent_level
        print(f"{print_indent}Processing for {group_name} ({len(files_in_group)} files)")

        yield from process_file_groups(
            files_in_group, remaining_criteria, model_name_for_log, group_path_names + (group_name,)
        )

# --- Main Analysis Logic ---
if __name__ == "__main__":

    dataset = "SimpleMC" # "SimpleQA" #
    game_type = "aop"#"dg" #
    output_entropy = False 
    USE_FILTERED_FOR_LOGRES = False #remove items where capabilites and game correctness disagree
    USE_ADJUSTED_FOR_LOGRES = False #use adjusted capabilities for logres

    LOG_FILENAME = f"analysis_log_multi_logres_{game_type}_{dataset.lower()}.txt"
    print(f"Loading main {dataset} dataset for features...")
    sqa_all_questions = load_and_format_dataset(dataset)
    sqa_feature_lookup = {
        item['id']: {
            'answer_type': item.get('answer_type', 0),
            'topic': item['topic'],
            'q_text': item['question']
        } for item in sqa_all_questions
    }
    print(f"sqa feature lookup created with {len(sqa_feature_lookup)} entries.")

    game_logs_dir = "./delegate_game_logs/" if game_type == "dg" else "./pass_game_logs/"
    capabilities_dir = "./compiled_results_sqa/" if dataset == "SimpleQA" else "./compiled_results_smc/"
    game_file_suffix = "_evaluated" if dataset == "SimpleQA" else ""

    if not os.path.isdir(game_logs_dir) or not os.path.isdir(capabilities_dir):
        print(f"Error: Ensure directories exist: {game_logs_dir}, {capabilities_dir}")
        exit()

    skip_files = None#["mistral-small-3.2-24b-instruct_SimpleMC_noqcnt_nopcnt_noscnt_temp0.0_1759095214_game_data.json"]
    hit_files = None#["mistral-small-3.2-24b-instruct_SimpleMC_noqcnt_nopcnt_noscnt_decisonOnly_temp0.0_1759249567_game_data.json"]

    model_game_files = defaultdict(list)
    for game_filename in sorted(os.listdir(game_logs_dir)):
        if skip_files and game_filename in skip_files:
            continue
        if hit_files and game_filename not in hit_files:
            continue

        if game_filename.endswith(f"_game_data{game_file_suffix}.json") and f"_{dataset}_" in game_filename:
            model_name_part = game_filename.split(f"_{dataset}_")[0]
            model_game_files[model_name_part].append(os.path.join(game_logs_dir, game_filename))

    subj_acc_override_pattern = re.compile(r"_subj\d+(\.\d+)?_")
    subj_game_override_pattern = re.compile(r"_subjgame\d+(\.\d+)?_")

    if game_type == "dg":
        FILE_GROUPING_CRITERIA = [
            {'name_prefix': "Feedback", 'split_logic': split_by_feedback},
            {'name_prefix': "Redaction", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_redacted_" in bn, "Redacted", "Non_Redacted")},
            {'name_prefix': "SubjAccOverride", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: subj_acc_override_pattern.search(bn), "SubjAccOverride", "NoSubjAccOverride")},
            {'name_prefix': "SubjGameOverride", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: subj_game_override_pattern.search(bn), "SubjGameOverride", "NoSubjGameOverride")},
            {'name_prefix': "Randomized", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_randomized_" in bn, "Randomized", "NotRandomized")},
            {'name_prefix': "NoHistory", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_nohistory_" in bn, "NoHistory", "WithHistory")},
    #        {'name_prefix': "Summary", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_summary_" in bn, "Summary", "NoSummary")},
            {'name_prefix': "Filtered", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_filtered_" in bn, "Filtered", "NotFiltered")},
            {'name_prefix': "decisionOnly", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_decisionOnly_" in bn, "decisionOnly", "decpluschoice")},
        ]
    else:
        FILE_GROUPING_CRITERIA = [
        {'name_prefix': "MsgHist", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_hist_" in bn, "MsgHist", "NoMsgHist")},
        {'name_prefix': "QCtr", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_noqcnt_" in bn, "NoQCtr", "QCtr")},
        {'name_prefix': "PCtr", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_nopcnt_" in bn, "NoPCtr", "PCtr")},
        {'name_prefix': "SCtr", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_noscnt_" in bn, "NoSCtr", "SCtr")},
        {'name_prefix': "decisionOnly", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_decisionOnly_" in bn, "decisionOnly", "decpluschoice")},
        ]
        
    entropy_rows, misused_results = [], []
    entropy_ofile = f"entropy_{game_type}_npcp_summary.csv"
    for model_name_part, game_files_for_model in model_game_files.items():
        print(f"\nProcessing model: {model_name_part} (total {len(game_files_for_model)} game files)")
        if not game_files_for_model:
            print(f"  No game files found for model {model_name_part}. Skipping.")
            continue

        for group_names_tuple, current_game_files_for_analysis in process_file_groups(
                game_files_for_model, FILE_GROUPING_CRITERIA, model_name_part):
            
            capabilities_filename = f"{model_name_part}_phase1_compiled.json"
            capabilities_file_path = os.path.join(capabilities_dir, capabilities_filename)

            if not os.path.exists(capabilities_file_path):
                print(f"{'  '*(len(group_names_tuple)+1)}Corresponding capabilities file not found: {capabilities_file_path}. Skipping this group.")
                continue

            s_i_map_for_this_model = {}
            p_i_map_for_this_model = {}
            a_i_map_for_this_model = {}
            entropy_map_for_this_model = {}
            try:
                with open(capabilities_file_path, 'r', encoding='utf-8') as f_cap:
                    cap_data = json.load(f_cap)
                for q_id, res_info in cap_data.get("results", {}).items():
                    if res_info.get("is_correct") is not None:
                        s_i_map_for_this_model[q_id] = 1 if res_info["is_correct"] else 0

                    probs_dict = res_info.get("probs")
                    subject_answer = res_info.get("subject_answer")
                    try: 
                        options = res_info['options'] if 'options' in res_info else res_info['question']['options'] if 'options' in res_info['question'] else None
                        subject_answer_full = options[subject_answer] if options and isinstance(options, dict) and subject_answer in options and game_type=="dg" else subject_answer
                    except Exception as e:
                        print(f"Error on capabilities qid {q_id}, res_info={res_info}: {e}")
                        raise e
                    a_i_map_for_this_model[q_id] = (subject_answer, subject_answer_full)
                    # Populate p_i_map_for_this_model
                    if subject_answer is not None and isinstance(probs_dict, dict):### and 'deepseek' not in model_name_part.lower():
                        prob_for_subject_answer = probs_dict.get(subject_answer)
                        if isinstance(prob_for_subject_answer, (int, float)):
                            if len(probs_dict.keys()) > 1:
                                p_i_map_for_this_model[q_id] = float(prob_for_subject_answer)
                            else:
                                p_i_map_for_this_model[q_id] = float(prob_for_subject_answer)**(1/max((len(subject_answer)//2),1)) #-math.log(float(prob_for_subject_answer))/max((len(subject_answer)//2),1)###float(prob_for_subject_answer)#approx token count
                    # Calculate and populate entropy_map_for_this_model
                    if isinstance(probs_dict, dict) and probs_dict:### and 'deepseek' not in model_name_part.lower():
                        prob_values = [float(p) for p in probs_dict.values() if isinstance(p, (int, float)) and p > 1e-9]
                        if prob_values:
                            entropy = -np.sum([p_val * np.log2(p_val) for p_val in prob_values if p_val > 1e-9]) if len(probs_dict.keys()) > 1 else  -math.log2(p_i_map_for_this_model[q_id])
                            entropy_map_for_this_model[q_id] = entropy

            except Exception as e:
                print(f"{'  '*(len(group_names_tuple)+1)}Error loading Capabilities file {capabilities_file_path}: {e}. Skipping this group.")
                continue

            print(f"{'  '*(len(group_names_tuple)+1)}Loaded responses for {len(s_i_map_for_this_model)} items.")
            o_map_for_this_model = {}
            from pathlib import Path
            capabilities_3p_file_path = next(Path("capabilities_3p_test_logs/").glob(f"{model_name_part}_{dataset}*.json"), None)
            try:
                with open(capabilities_3p_file_path, 'r', encoding='utf-8') as f_cap:
                    cap_data = json.load(f_cap)
                for q_id, res_info in cap_data.get("results", {}).items():
                    o_map_for_this_model[q_id] = res_info.get("is_correct")
            except Exception as e:
                print(f"{'  '*(len(group_names_tuple)+1)}Error loading 3P Capabilities file {capabilities_3p_file_path}: {e}.")

            sp_map_for_this_model = {}
            capabilities_1p_file_path = next(Path("capabilities_1p_test_logs/").glob(f"{model_name_part}_{dataset}*.json"), None)
            print("capabilities_1p_file_path=", capabilities_1p_file_path)
            try:
                with open(capabilities_1p_file_path, 'r', encoding='utf-8') as f_cap:
                    cap_data = json.load(f_cap)
                for q_id, res_info in cap_data.get("results", {}).items():
                    sp_map_for_this_model[q_id] = res_info.get("is_correct")
            except Exception as e:
                print(f"{'  '*(len(group_names_tuple)+1)}Error loading 1P Capabilities file {capabilities_1p_file_path}: {e}.")

            if not s_i_map_for_this_model:
                print(f"{'  '*(len(group_names_tuple)+1)}No S_i data loaded from {capabilities_file_path}. Skipping this group.")
                continue

            if not current_game_files_for_analysis: # Should be caught by process_file_groups, but as a safeguard
                print(f"{'  '*(len(group_names_tuple)+1)}No game files for analysis for this group. Skipping.")
                continue
            
            df_model, subject_acc_phase1, phase2_corcnt, phase2_totalcnt, teammate_acc_phase1 = prepare_regression_data_for_model(current_game_files_for_analysis,
                                                         sqa_feature_lookup,
                                                         s_i_map_for_this_model,
                                                         a_i_map_for_this_model,
                                                         p_i_map_for_this_model,
                                                         entropy_map_for_this_model,
                                                         o_map_for_this_model,
                                                         sp_map_for_this_model,
                                                         game_file_suffix=game_file_suffix)

            if df_model is None or df_model.empty:
                print(f"{'  '*(len(group_names_tuple)+1)}No data for regression analysis for group: {model_name_part} ({', '.join(group_names_tuple)}).")
                continue

            if 'teammate_skill_ratio' in df_model.columns:
                mean_skill_ratio = df_model['teammate_skill_ratio'].mean()
                df_model['teammate_skill_ratio'] = df_model['teammate_skill_ratio'] - mean_skill_ratio
            
            log_context_str = f"{model_name_part} ({', '.join(group_names_tuple)}, {len(current_game_files_for_analysis)} game files)"

#            if not ((game_type == "dg" and "Feedback_False, Non_Redacted, NoSubjAccOverride, NoSubjGameOverride, NotRandomized, WithHistory, NotFiltered" in log_context_str) or (game_type == "aop" and "NoMsgHist, NoQCtr, NoPCtr, NoSCtr" in log_context_str)): continue
            if not ((game_type == "dg" and "Feedback_False, Non_Redacted, NoSubjAccOverride, NoSubjGameOverride, NotRandomized" in log_context_str) or (game_type == "aop" and "NoMsgHist, NoQCtr, NoPCtr, NoSCtr" in log_context_str)): continue

            log_output(f"\n--- Analyzing {log_context_str} ---", print_to_console=True, suppress=False)
            log_output(f"              Game files for analysis: {current_game_files_for_analysis}\n", suppress=False)
            if game_type == "dg": log_output(f"Teammate accuracy phase 1: {teammate_acc_phase1:.4f}", suppress=False)
            
            try:
                log_output(f"Delegation rate: {df_model['delegate_choice'].mean():.4f} (n={len(df_model)})", suppress=False)
                log_output(f"df_model['delegate_choice'].value_counts()= {df_model['delegate_choice'].value_counts(dropna=False)}\n")
                if 's_i_capability' in df_model.columns:
                    cross_tab_s_i = pd.crosstab(df_model['delegate_choice'], df_model['s_i_capability'])
                    #TP = cross_tab_s_i.loc[1, 0]; FP = cross_tab_s_i.loc[1, 1]; FN = cross_tab_s_i.loc[0, 0]; TN = cross_tab_s_i.loc[0, 1]
                    delegated = np.array(df_model['delegate_choice'], bool)
                    kept_mask = ~delegated                       # True where model answered itself
                    cap_corr = np.array(df_model['s_i_capability'], bool)   # Baseline correctness from capabilities file
                    team_corr = np.where(df_model['delegate_choice'] == 0, df_model['subject_correct'].fillna(0).astype(bool), False) #Real in-game self correctness (only defined when kept)
                    try:
                        N = len(df_model)
                        if teammate_acc_phase1:
                            p_const = max(subject_acc_phase1, teammate_acc_phase1) 
                            team_correct = np.array(df_model['team_correct'], int)
                            lo, hi  = proportion_confint(team_correct.sum(), N, method="wilson")
                            excess  = team_correct.mean() - p_const
                        else:#pass game
                            p_const = max(cap_corr.mean(), 0.5)
                            Cor = team_corr.astype(int).sum()
                            pass_mask = df_model["delegate_choice"] == 1   # passed
                            P = pass_mask.sum()
                            p_team = (Cor + 0.5 * P) / N
                            se = math.sqrt(p_team * (1 - p_team) / N)    # Wald SE
                            lo = p_team - 1.96 * se
                            hi = p_team + 1.96 * se
                            excess = p_team - p_const
                        ci_excess = (lo - p_const, hi - p_const)
                        log_output(f"Team-acc: {excess+p_const}", suppress=False)
                        log_output(f"Team-acc lift = {excess:.3f} [{ci_excess[0]:.3f}, {ci_excess[1]:.3f}]", suppress=False)
                        res_dicts[model_name_part]['team_acc_lift'] = {'lift': excess, 'ci_lo': ci_excess[0], 'ci_hi': ci_excess[1]}
                    except Exception as e:
                        log_output(f"Error calculating team-acc lift: {e}")

                    if 'p_i_capability' in df_model.columns and df_model['p_i_capability'].notna().any():
                        try:
                            team_acc = teammate_acc_phase1 if teammate_acc_phase1 else cap_corr.mean()
                            res = compute_optimal_accuracy_with_introspection(df_model['p_i_capability'], df_model['s_i_capability'], df_model['delegate_choice'], team_acc)
                            log_output(f"Max possible team accuracy: {res['optimal_accuracy']:.4f}", suppress=False)
                            log_output(f"Optimal delegation rate: {(1-res['optimal_answer_rate']):.4f}", suppress=False)
                            log_output(f"Best possible self-accuracy: {res['accuracy_on_answered']:.4f}", suppress=False)
                            log_output(f"Agreement rate: {res['agreement_rate']:.4f} [{res['agreement_rate_ci'][0]:.4f}, {res['agreement_rate_ci'][1]:.4f}]", suppress=False)
                            log_output(f"Weighted agreement rate: {res['weighted_agreement_rate']:.4f}", suppress=False)
                            log_output(f"Norm-weighted agreement rate: {res['norm_weighted_agreement_rate']:.4f}", suppress=False)
                            log_output(f"Underconfidence rate: {res['underconf_rate']:.4f}", suppress=False)
                            log_output(f"Overconfidence rate: {res['overconf_rate']:.4f}", suppress=False)
                            log_output(f"Weighted underconfidence rate: {res['weighted_underconf_rate']:.4f}", suppress=False)
                            log_output(f"Weighted overconfidence rate: {res['weighted_overconf_rate']:.4f}", suppress=False)
                            log_output(f"Unweighted confidence: {res['unweighted_confidence']:.4f} [{res['unweighted_confidence_ci'][0]:.4f}, {res['unweighted_confidence_ci'][1]:.4f}]", suppress=False)
                            log_output(f"Weighted confidence: {res['weighted_confidence']:.4f} [{res['weighted_confidence_ci'][0]:.4f}, {res['weighted_confidence_ci'][1]:.4f}]", suppress=False)
                        except Exception as e:
                            log_output(f"Error calculating optimal accuracy with introspection: {e}", suppress=False)

                    # Hybrid correctness label 
                    #    – use real game correctness when the model kept
                    #    – fallback to baseline correctness when it delegated
                    true_label = np.where(kept_mask, team_corr, cap_corr)   # 1 = model would be correct

                    #mcc, score, ci = mcc_ci_boot(TP=TP, FN=FN, FP=FP, TN=TN)
                    TP, FN, FP, TN = contingency(delegated, cap_corr) #(TP=delegated, wrong in baseline, FP=delegated, correct in baseline, FN=answered, wrong in baseline, TN=answered, correct in baseline)
                    Recall = TN / (TN + FP)
                    Precision = TN / (TN + FN)
                    try: 
                        raw_stats = lift_mcc_stats(TP, FN, FP, TN, team_corr[kept_mask], cap_corr.mean(), baseline_correct=df_model['s_i_capability'], delegated=df_model['delegate_choice'], baseline_probs = df_model['p_i_capability'] if 'p_i_capability' in df_model.columns and df_model['p_i_capability'].notna().any() else None)
                        raw_stats_ent = lift_mcc_stats(TP, FN, FP, TN, team_corr[kept_mask], cap_corr.mean(), baseline_correct=1-df_model['s_i_capability'], delegated=df_model['delegate_choice'], baseline_probs = df_model['capabilities_entropy'] if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any() else None)
                    except Exception as e:
                        log_output(f"Error calculating raw_stats: {e}", suppress=False)
                    log_output(f"Introspection score = {raw_stats['mcc']:.3f} [{raw_stats['mcc_ci'][0]:.3f}, {raw_stats['mcc_ci'][1]:.3f}], p={raw_stats['p_mcc']:.4g}", suppress=False)
                    res_dicts[model_name_part]['introspection_score'] = {'mcc': raw_stats['mcc'], 'p': raw_stats['p_mcc']}
                    log_output(f"FP = {FP}", suppress=False)
                    log_output(f"FN = {FN}", suppress=False)

                    log_output(f"Cross-tabulation of delegate_choice vs. s_i_capability:\n{cross_tab_s_i}\n")
                    prob_delegating_Si0 = df_model.loc[df_model['s_i_capability'] == 0, 'delegate_choice'].mean()
                    log_output(f"Probability of delegating when s_i_capability is 0: {prob_delegating_Si0:.4f}", suppress=False)
                    prob_delegating_Si1 = df_model.loc[df_model['s_i_capability'] == 1, 'delegate_choice'].mean()
                    log_output(f"Probability of delegating when s_i_capability is 1: {prob_delegating_Si1:.4f}", suppress=False)
                    log_output(f"Phase 1 accuracy: {cap_corr.mean():.4f} (n={len(cap_corr)})", suppress=False)
                    #log_output(f"Phase 1 accuracy: {subject_acc_phase1:.4f} (n=400)")
                    log_output(f"Phase 2 self-accuracy: {phase2_corcnt/phase2_totalcnt:.4f} (n={phase2_totalcnt})", suppress=False)
                
                    #lift_sub, ci_low, ci_high, p_boot = self_acc_stats(cap_corr, team_corr, kept_mask)
                    log_output(f"Self-acc lift = {raw_stats['lift']:.3f} [{raw_stats['lift_ci'][0]:.3f}, {raw_stats['lift_ci'][1]:.3f}], p={raw_stats['p_lift']:.4g}", suppress=False)
                    log_output(f"Normed Self-acc lift = {raw_stats['normed_lift']:.3f} [{raw_stats['normed_lift_ci'][0]:.3f}, {raw_stats['normed_lift_ci'][1]:.3f}]", suppress=False)
                    res_dicts[model_name_part]['self_acc_lift'] = {'lift': raw_stats['lift'], 'p': raw_stats['p_lift']}
                    res_dicts[model_name_part]['normed_self_acc_lift'] = {'lift': raw_stats['normed_lift'], 'ci_lo': raw_stats['normed_lift_ci'][0], 'ci_hi': raw_stats['normed_lift_ci'][1]}
                    log_output(f"Balanced Accuracy Effect Size = {raw_stats['single_point_auc']:.3f} [{raw_stats['single_point_auc_ci'][0]:.3f}, {raw_stats['single_point_auc_ci'][1]:.3f}]", suppress=False)
                    res_dicts[model_name_part]['single_point_auc'] = {'auc': raw_stats['single_point_auc'], 'ci_lo': raw_stats['single_point_auc_ci'][0], 'ci_hi': raw_stats['single_point_auc_ci'][1]}
                    if raw_stats['full_auc'] is not None:
                        log_output(f"Full AUC = {raw_stats['full_auc']:.3f} [{raw_stats['full_auc_ci'][0]:.3f}, {raw_stats['full_auc_ci'][1]:.3f}]", suppress=False)
                        res_dicts[model_name_part]['full_auc'] = {'auc': raw_stats['full_auc'], 'ci_lo': raw_stats['full_auc_ci'][0], 'ci_hi': raw_stats['full_auc_ci'][1]}
                        log_output(f"Calibration AUC = {raw_stats['calibration_auc']:.3f} [{raw_stats['calibration_auc_ci'][0]:.3f}, {raw_stats['calibration_auc_ci'][1]:.3f}]", suppress=False)
                        res_dicts[model_name_part]['calibration_auc'] = {'auc': raw_stats['calibration_auc'], 'ci_lo': raw_stats['calibration_auc_ci'][0], 'ci_hi': raw_stats['calibration_auc_ci'][1]}
                        log_output(f"Calibration Entropy AUC = {raw_stats_ent['calibration_auc']:.3f} [{raw_stats_ent['calibration_auc_ci'][0]:.3f}, {raw_stats_ent['calibration_auc_ci'][1]:.3f}]", suppress=False)
                        res_dicts[model_name_part]['calibration_entropy_auc'] = {'auc': raw_stats_ent['calibration_auc'], 'ci_lo': raw_stats_ent['calibration_auc_ci'][0], 'ci_hi': raw_stats_ent['calibration_auc_ci'][1]}

                if 's_i_capability' in df_model.columns: # self_correct is s_i_capability when delegate_choice == 0
                    self_choice_df = df_model[df_model['delegate_choice'] == 0]
                    if not self_choice_df.empty:
                        cross_tab_self_s_i_vs_team = pd.crosstab(self_choice_df['s_i_capability'], self_choice_df['subject_correct'])
                        TP = cross_tab_self_s_i_vs_team.loc[1, False]; FP = cross_tab_self_s_i_vs_team.loc[1, True]; FN = cross_tab_self_s_i_vs_team.loc[0, False]; TN = cross_tab_self_s_i_vs_team.loc[0, True]
                        cr = len(self_choice_df[self_choice_df['answer_changed'] == True]) / len(self_choice_df)
                        gcr = len(self_choice_df[(self_choice_df['subject_correct'] == True) & (self_choice_df['answer_changed'] == True)]) / len(self_choice_df)
                        bcr = len(self_choice_df[(self_choice_df['s_i_capability'] == 1) & (self_choice_df['subject_correct'] == False) & (self_choice_df['answer_changed'] == True)]) / len(self_choice_df)
                        b2bcr = len(self_choice_df[(self_choice_df['s_i_capability'] == 0) & (self_choice_df['subject_correct'] == False) & (self_choice_df['answer_changed'] == True)]) / len(self_choice_df)
                        log_output(f"Game-Test Change Rate: {cr:.4f}", suppress=False)
                        log_output(f"Game-Test Good Change Rate: {gcr:.4f}", suppress=False)
                        log_output(f"Game-Test Bad Change Rate: {bcr:.4f}", suppress=False)
                        log_output(f"Game-Test Bad-to-Bad Change Rate: {b2bcr:.4f}", suppress=False)

                        weird_cases = self_choice_df[(self_choice_df['answer_changed'] == True) & (self_choice_df['s_i_capability'] == 1) & (self_choice_df['subject_correct'] == True)]
                        log_output(f"Number of answer changes without correctness flip: {len(weird_cases)}", suppress=False)
                        if len(weird_cases)>0: log_output(weird_cases[['s_i_capability', 'subject_correct', 'answer_changed', 'q_id']].head(), suppress=False)

                        changes_df = self_choice_df[self_choice_df['answer_changed'] == True]
                        if len(changes_df) > 0:
                            good_given_change = len(changes_df[changes_df['subject_correct'] == True]) / len(changes_df)
                            bad_given_change = len(changes_df[(changes_df['s_i_capability'] == 1) & (changes_df['subject_correct'] == False)]) / len(changes_df)
                            log_output(f"P(good | change): {good_given_change:.4f}", suppress=False)
                            log_output(f"P(bad | change): {bad_given_change:.4f}", suppress=False)

                        cir = len(self_choice_df[(self_choice_df['s_i_capability'] == 0) & (self_choice_df['answer_changed'] == True)]) / len(self_choice_df[self_choice_df['s_i_capability'] == 0])
                        log_output(f"Game-Test Change on Incor Rate: {cir}", suppress=False)
                        cir = len(self_choice_df[(self_choice_df['s_i_capability'] == 1) & (self_choice_df['answer_changed'] == True)]) / len(self_choice_df[self_choice_df['s_i_capability'] == 1])
                        log_output(f"Game-Test Change on Cor Rate: {cir}", suppress=False)

                        if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any() and 'normalized_prob_entropy' in df_model.columns and df_model['normalized_prob_entropy'].notna().any():
                            #compute correlation between capabilities_entropy and normalized_prob_entropy with confidence intervals
                            corr, p_val = pearsonr(self_choice_df['capabilities_entropy'], self_choice_df['normalized_prob_entropy'])
                            n = len(self_choice_df)
                            stderr = 1.0 / math.sqrt(n - 3)
                            delta = 1.96 * stderr
                            atanh_corr = np.arctanh(corr)
                            ci_low = np.tanh(atanh_corr - delta)
                            ci_high = np.tanh(atanh_corr + delta)
                            log_output(f"Capent-Gament corr: {corr:.4f} [{ci_low:.4f}, {ci_high:.4f}], p={p_val:.4g}", suppress=False)

                log_output("\n  Model 1: Delegate_Choice ~ S_i_capability")
                try:
                    logit_model1 = smf.logit('delegate_choice ~ s_i_capability', data=df_model).fit(disp=0)
                    log_output(logit_model1.summary(), suppress=False)
                except Exception as e_full:
                    log_output(f"                    Could not fit Model 1: {e_full}", suppress=False)

                if 'o_prob' in df_model.columns and df_model['o_prob'].notna().any():
                    log_output("\n  Model 1.3: Delegate_Choice ~ Other's Prob")
                    try:
                        logit_m2 = smf.logit('delegate_choice ~ o_prob', data=df_model.dropna(subset=['o_prob', 'delegate_choice'])).fit(disp=0)
                        log_output(logit_m2.summary())
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 1.3: {e_full}")

                if 'sp_prob' in df_model.columns and df_model['sp_prob'].notna().any():
                    log_output("\n  Model 1.31: Delegate_Choice ~ Self Prob")
                    try:
                        logit_m2 = smf.logit('delegate_choice ~ sp_prob', data=df_model.dropna(subset=['sp_prob', 'delegate_choice'])).fit(disp=0)
                        log_output(logit_m2.summary())
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 1.31: {e_full}")


                    implicit_prob_str = 'capabilities_entropy' #'p_i_capability' # 
                    """
                    results = compare_predictors_of_answer(
                        np.array(df_model['sp_prob']),
                        np.array(df_model[implicit_prob_str]),
                        np.array(df_model['delegate_choice'])
                    )
                    log_output("\n  Predicting delegate_choice")
                    log_output(f"Stated AUC: {results['auc_stated']:.3f}")
                    log_output(f"Implicit AUC: {results['auc_implicit']:.3f}")
                    log_output(f"Combined AUC: {results['auc_both']:.3f}")
                    log_output(f"Standardized coef (stated): {results['coef_stated']:.3f}")
                    log_output(f"Standardized coef (implicit): {results['coef_implicit']:.3f}")
                    log_output(f"p_implicit_adds_to_stated: {results['p_implicit_adds_to_stated']:.4f}")
                    log_output(f"p_stated_adds_to_implicit: {results['p_stated_adds_to_implicit']}")
                    better_standalone = 'stated' if results['auc_stated'] > results['auc_implicit'] else 'implicit'
                    better_in_combined = 'stated' if abs(results['coef_stated']) > abs(results['coef_implicit']) else 'implicit'
                    log_output(f"better_standalone: {better_standalone}")
                    log_output(f"better_in_combined: {better_in_combined}")

                    results = compare_predictors_of_answer(
                        np.array(df_model['sp_prob']),
                        np.array(df_model[implicit_prob_str]),
                        np.array(df_model['s_i_capability'])
                    )
                    log_output("\nPredicting baseline correctness (s_i_capability):")
                    log_output(f"Stated AUC: {results['auc_stated']:.3f}")
                    log_output(f"Implicit AUC: {results['auc_implicit']:.3f}")
                    log_output(f"Combined AUC: {results['auc_both']:.3f}")
                    log_output(f"Standardized coef (stated): {results['coef_stated']:.3f}")
                    log_output(f"Standardized coef (implicit): {results['coef_implicit']:.3f}")
                    log_output(f"p_implicit_adds_to_stated: {results['p_implicit_adds_to_stated']:.4f}")
                    log_output(f"p_stated_adds_to_implicit: {results['p_stated_adds_to_implicit']}")
                    better_standalone = 'stated' if results['auc_stated'] > results['auc_implicit'] else 'implicit'
                    better_in_combined = 'stated' if abs(results['coef_stated']) > abs(results['coef_implicit']) else 'implicit'
                    log_output(f"better_standalone: {better_standalone}")
                    log_output(f"better_in_combined: {better_in_combined}")

                    if 'o_prob' in df_model.columns and df_model['o_prob'].notna().any():
                        results = compare_predictors_of_answer(
                            np.array(df_model['sp_prob']),
                            np.array(df_model['o_prob']),
                            np.array(df_model['delegate_choice'])
                        )                    
                        log_output(f"Self AUC: {results['auc_stated']:.3f}")
                        log_output(f"Other AUC: {results['auc_implicit']:.3f}")
                        log_output(f"Combined AUC: {results['auc_both']:.3f}")
                        log_output(f"Standardized coef (self): {results['coef_stated']:.3f}")
                        log_output(f"Standardized coef (other): {results['coef_implicit']:.3f}")
                        log_output(f"p_other_adds_to_self: {results['p_implicit_adds_to_stated']:.4f}")
                        log_output(f"p_self_adds_to_other: {results['p_stated_adds_to_implicit']}")
                        better_standalone = 'self' if results['auc_stated'] > results['auc_implicit'] else 'other'
                        better_in_combined = 'self' if abs(results['coef_stated']) > abs(results['coef_implicit']) else 'other'
                        log_output(f"better_standalone: {better_standalone}")
                        log_output(f"better_in_combined: {better_in_combined}")

                    if game_type == "aop":
                        results = compare_predictors_of_implicit_conf((df_model['sp_prob'] < 0.5).astype(int), df_model['delegate_choice'],df_model[implicit_prob_str])
                    else:
                        results = compare_predictors_of_implicit_conf((df_model['sp_prob'] < df_model['teammate_skill']).astype(int), df_model['delegate_choice'],df_model[implicit_prob_str])
                    log_output(f"\nActual pass correlation: {results['corr_actual']:.3f} (p={results['p_actual']:.4f})")
                    log_output(f"Stated pass correlation: {results['corr_stated']:.3f} (p={results['p_stated']:.4f})") 
                    log_output(f"Actual-Stated Difference: p={results['p_diff']:.4f}")
                    #now leave as continuous
                    results = compare_predictors_of_implicit_conf(np.array([1-p for p in df_model['sp_prob']]), df_model['delegate_choice'],df_model[implicit_prob_str])
                    log_output(f"Stated pass continuous correlation: {results['corr_stated']:.3f} (p={results['p_stated']:.4f})") 
                    log_output(f"Actual-Continuous Stated Difference: p={results['p_diff']:.4f}")
                    """

                if len(df_model) > 20 :
                    min_obs_per_category=int(len(df_model)/15) + 1
#                    if 'grok' in model_name_part or '3-sonnet' in model_name_part or '3-haiku' in model_name_part: min_obs_per_category = 100
                    
                    if 'o_prob' in df_model.columns and df_model['o_prob'].notna().any() and 'sp_prob' in df_model.columns and df_model['sp_prob'].notna().any():
                        log_output(f"\nCorrelation between Other's Prob and Self Prob: {df_model['o_prob'].corr(df_model['sp_prob'])}", suppress=False)
                        log_output(f"\nCorrelation between {implicit_prob_str} and Self Prob: {df_model[implicit_prob_str].corr(df_model['sp_prob'])}", suppress=False)
                        df_model['sp_binary'] = (df_model['sp_prob'] >= 0.5).astype(int)

                    topic_column_for_formula = 'topic_grouped' if 'topic_grouped' in df_model else 'topic'
                    ans_type_column_for_formula = 'answer_type_grouped' if 'answer_type_grouped' in df_model else 'answer_type'

                    base_model_terms = [
                        's_i_capability',
                        f'C({topic_column_for_formula})',
                        f'C({ans_type_column_for_formula})',
                        'q_length',
                    ]

                    conditional_regressors = ['summary', 'nobio', 'noeasy', 'noctr', 'judge_delegate', 'teammate_judge_delegate']

                    final_model_terms = list(base_model_terms)
                    if 'teammate_skill_ratio' in df_model.columns:
                        final_model_terms.append('teammate_skill_ratio')
                        final_model_terms.append(f"s_i_capability:teammate_skill_ratio") # Interaction term
                    for regressor in conditional_regressors:
                        if regressor in df_model.columns:
                            final_model_terms.append(regressor)
                    
                    fit_kwargs = {'disp': 0}
                    if df_model['q_id'].duplicated().any():
                        fit_kwargs.update({'cov_type': 'cluster', 'cov_kwds': {'groups': df_model['q_id']}})
                        log_output("                    Model 4: Using clustered standard errors by q_id.")
                            
                    model_def_str_4 = 'delegate_choice ~ ' + ' + '.join(final_model_terms)
                    interaction_str = "Interactions" if 's_i_capability:teammate_skill_ratio' in final_model_terms else "No Interactions"
                    log_output(f"\n                  Model 4 ({interaction_str}): {model_def_str_4}", suppress=False)
                    try:
                        logit_model4 = smf.logit(model_def_str_4, data=df_model).fit(**fit_kwargs)
                        res_dicts[model_name_part]['s_i_capability'] = {'coef': float(logit_model4.params['s_i_capability']),'p': float(logit_model4.pvalues['s_i_capability'])}
                        log_output(logit_model4.summary(), suppress=False)
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 4: {e_full}", suppress=False)
                        logit_model4, final_terms, removed = remove_collinear_terms(final_model_terms, df_model, 'delegate_choice', fit_kwargs, protected_terms=['s_i_capability'])
                        res_dicts[model_name_part]['s_i_capability'] = {'coef': float(logit_model4.params['s_i_capability']),'p': float(logit_model4.pvalues['s_i_capability'])}
                        log_output(logit_model4.summary(), suppress=False)

                    try:
                        continuous_controls = [df_model[t] for t in final_model_terms if t not in ['s_i_capability', 's_i_capability:teammate_skill_ratio', 'teammate_skill_ratio'] and not (isinstance(t, str) and t.startswith('C('))]
                        categorical_controls = [df_model[t.replace('C(', '').replace(')', '')] for t in final_model_terms if (isinstance(t, str) and t.startswith('C('))]
                        control_vars = continuous_controls + categorical_controls 

                        res = partial_correlation_on_decision(dv_series=1-df_model['delegate_choice'], iv_series=df_model['s_i_capability'], control_series_list=continuous_controls+categorical_controls)
                        log_output(f"Partial correlation on decision with Correctness, surface controls: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)

                        df_answer_unchanged = df_model[(df_model['answer_changed'] == False) | (df_model['delegate_choice'] == 1)]
                        continuous_controls_sa = [df_answer_unchanged[t] for t in final_model_terms if t not in ['s_i_capability', 's_i_capability:teammate_skill_ratio', 'teammate_skill_ratio'] and not (isinstance(t, str) and t.startswith('C('))]
                        categorical_controls_sa = [df_answer_unchanged[t.replace('C(', '').replace(')', '')] for t in final_model_terms if (isinstance(t, str) and t.startswith('C('))]
                        control_vars_sa = continuous_controls + categorical_controls 
                        res = partial_correlation_on_decision(dv_series=1-df_answer_unchanged['delegate_choice'], iv_series=df_answer_unchanged['s_i_capability'], control_series_list=control_vars_sa)
                        log_output(f"Partial correlation on decision with Correctness for same answer, surface controls: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)

                    except Exception as e_full:
                        log_output(f"                    Could not fit Logit on decision with correctness and surface controls: {e_full}", suppress=False)

                    if 'o_prob' in df_model.columns and df_model['o_prob'].notna().any():
                        try:
                            res = logit_on_decision(pass_decision=1-df_model['delegate_choice'], iv_of_interest=df_model['s_i_capability'], control_vars=[df_model['o_prob']])
                            ci_lower, ci_upper = res.conf_int().loc['s_i_capability']
                            log_output(f"Baseline correctness coefficient with Stated Other control: {res.params['s_i_capability']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], z={res.tvalues['s_i_capability']:.4f}, stderr={res.bse['s_i_capability']:.4f}",suppress=False)
                            res = logit_on_decision(pass_decision=1-df_model['delegate_choice'], iv_of_interest=df_model['s_i_capability'], control_vars=control_vars + [df_model['o_prob']])
                            ci_lower, ci_upper = res.conf_int().loc['s_i_capability']
                            log_output(f"Baseline correctness coefficient with all controls: {res.params['s_i_capability']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], z={res.tvalues['s_i_capability']:.4f}, stderr={res.bse['s_i_capability']:.4f}",suppress=False)
                            z, z_ci_low, z_ci_high = (res.params['s_i_capability']/res.bse['s_i_capability'],) + tuple((res.conf_int().loc['s_i_capability'] / res.bse['s_i_capability']).values)
                            log_output(f"Baseline correctness coefficient with all controls, standardized: {z:.4f} [{z_ci_low:.4f}, {z_ci_high:.4f}]", suppress=False)
                            res = logit_on_decision(pass_decision=1-df_model['delegate_choice'], iv_of_interest=None, control_vars=control_vars + [df_model['o_prob']])

                            res = partial_correlation_on_decision(dv_series=1-df_model['delegate_choice'], iv_series=df_model['s_i_capability'], control_series_list=[df_model['o_prob']])
                            log_output(f"Partial correlation on decision with Correctness, Stated Other control: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)
                            try:
                                res = partial_correlation_on_decision(dv_series=1-df_model['delegate_choice'], iv_series=df_model['s_i_capability'], control_series_list=[df_model['o_prob']]+continuous_controls+categorical_controls)
                                log_output(f"Partial correlation on decision with Correctness, all controls: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)
                            except Exception as e_full:
                                log_output(f"                    Could not fit partial correlation on decision with Correctness, all controls: {e_full}", suppress=False)
                            #log_output(f"res={res}", suppress=False)

                            res = partial_correlation_on_decision(dv_series=1-df_answer_unchanged['delegate_choice'], iv_series=df_answer_unchanged['s_i_capability'], control_series_list=[df_answer_unchanged['o_prob']])
                            log_output(f"Partial correlation on decision with Correctness for same answer, Stated Other control: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)
                            res = partial_correlation_on_decision(dv_series=1-df_answer_unchanged['delegate_choice'], iv_series=df_answer_unchanged['s_i_capability'], control_series_list=[df_answer_unchanged['o_prob']]+control_vars_sa)
                            log_output(f"Partial correlation on decision with Correctness for same answer, all controls: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)

                            res = logit_on_decision(pass_decision=1-df_model['delegate_choice'], iv_of_interest=df_model['capabilities_entropy'], control_vars=control_vars + [df_model['o_prob']])
                            ci_lower, ci_upper = res.conf_int().loc['capabilities_entropy']
                            log_output(f"Capent coefficient with all controls: {res.params['capabilities_entropy']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], z={res.tvalues['capabilities_entropy']:.4f}, stderr={res.bse['capabilities_entropy']:.4f}",suppress=False)
                            log_output(res.summary())
                            res = logit_on_decision(pass_decision=1-df_model['delegate_choice'], iv_of_interest=None, control_vars=control_vars + [df_model['o_prob']])
                            log_output(f"pseudo-R2, all controls model: {res.prsquared:.4f}", suppress=False)
                        except Exception as e_full:
                            log_output(f"                    Could not fit Logit on decision with correctness and Stated Other control: {e_full}", suppress=False)


                    # Model 5 (No interaction)
                    final_model_terms_m5 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t)]
                    model_def_str_5 = 'delegate_choice ~ ' + ' + '.join(final_model_terms_m5)
                    if model_def_str_5 != model_def_str_4:
                        log_output(f"\n                  Model 4 (No Interactions): {model_def_str_5}", suppress=False)
                        try:
                            logit_model5 = smf.logit(model_def_str_5, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model5.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 5: {e_full}", suppress=False)

                    try: 
                        res = block_partial_controls_given_entropy(dv_series=df_model['delegate_choice'], entropy_series=df_model['s_i_capability'], control_series_list=continuous_controls+categorical_controls)
                        log_output(f"Partial correlation on decision with surface controls, controlling for baseline correctness: {res['R_controls_given_entropy']:.4f} [{res['R_CI'][0]:.4f}, {res['R_CI'][1]:.4f}]", suppress=False)
                        if 'o_prob' in df_model.columns and df_model['o_prob'].notna().any():
                            res = block_partial_controls_given_entropy(dv_series=df_model['delegate_choice'], entropy_series=df_model['s_i_capability'], control_series_list=[df_model['o_prob']])
                            log_output(f"Partial correlation on decision with Stated Other control, controlling for baseline correctness: {res['R_controls_given_entropy']:.4f} [{res['R_CI'][0]:.4f}, {res['R_CI'][1]:.4f}]", suppress=False)
                            res = block_partial_controls_given_entropy(dv_series=df_model['delegate_choice'], entropy_series=df_model['s_i_capability'], control_series_list=[df_model['o_prob']]+continuous_controls+categorical_controls)
                            log_output(f"Partial correlation on decision with all controls, controlling for baseline correctness: {res['R_controls_given_entropy']:.4f} [{res['R_CI'][0]:.4f}, {res['R_CI'][1]:.4f}]", suppress=False)
                    except Exception as e_full:
                        log_output(f"                    Could not compute control cues partial correlation on correctness: {e_full}", suppress=False)
                        
                    if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any():
                        log_output(f"\n====================Partial correlation on decision: Capent====================")
                        try:
                            res = partial_correlation_on_decision(dv_series=df_model['delegate_choice'], iv_series=df_model['capabilities_entropy'], control_series_list=[])
                            log_output(f"Partial correlation on decision with Capent, no controls: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)
                            res = partial_correlation_on_decision(dv_series=df_model['delegate_choice'], iv_series=df_model['capabilities_entropy'], control_series_list=continuous_controls+categorical_controls)
                            log_output(f"Partial correlation on decision with Capent, surface controls: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)
                            if 'o_prob' in df_model.columns and df_model['o_prob'].notna().any():
                                res = partial_correlation_on_decision(dv_series=df_model['delegate_choice'], iv_series=df_model['capabilities_entropy'], control_series_list=[df_model['o_prob']])
                                log_output(f"Partial correlation on decision with Capent, Stated Other control: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)
                                res = partial_correlation_on_decision(dv_series=df_model['delegate_choice'], iv_series=df_model['capabilities_entropy'], control_series_list=[df_model['o_prob']]+continuous_controls+categorical_controls)
                                log_output(f"Partial correlation on decision with Capent, all controls: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)

                            res = partial_correlation_on_decision(dv_series=df_answer_unchanged['delegate_choice'], iv_series=df_answer_unchanged['capabilities_entropy'], control_series_list=continuous_controls_sa+categorical_controls_sa)
                            log_output(f"Partial correlation on decision with Capent, same answer, surface controls: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)
                            if 'o_prob' in df_model.columns and df_model['o_prob'].notna().any():
                                res = partial_correlation_on_decision(dv_series=df_answer_unchanged['delegate_choice'], iv_series=df_answer_unchanged['capabilities_entropy'], control_series_list=[df_answer_unchanged['o_prob']]+continuous_controls_sa+categorical_controls_sa)
                                log_output(f"Partial correlation on decision with Capent, same answer, all controls: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)

                            try:
                                res = block_partial_controls_given_entropy(dv_series=df_model['delegate_choice'], entropy_series=df_model['capabilities_entropy'], control_series_list=continuous_controls+categorical_controls)
                                log_output(f"Partial correlation on decision with surface controls: {res['R_controls_given_entropy']:.4f} [{res['R_CI'][0]:.4f}, {res['R_CI'][1]:.4f}]", suppress=False)
                            except Exception as e_full:
                                log_output(f"                    Could not compute partial correlation on capent with surface controls: {e_full}", suppress=False)
                            if 'o_prob' in df_model.columns and df_model['o_prob'].notna().any():
                                res = block_partial_controls_given_entropy(dv_series=df_model['delegate_choice'], entropy_series=df_model['capabilities_entropy'], control_series_list=[df_model['o_prob']])
                                log_output(f"Partial correlation on decision with Stated Other control: {res['R_controls_given_entropy']:.4f} [{res['R_CI'][0]:.4f}, {res['R_CI'][1]:.4f}]", suppress=False)
                                res = block_partial_controls_given_entropy(dv_series=df_model['delegate_choice'], entropy_series=df_model['capabilities_entropy'], control_series_list=[df_model['o_prob']]+continuous_controls+categorical_controls)
                                log_output(f"Partial correlation on decision with all controls: {res['R_controls_given_entropy']:.4f} [{res['R_CI'][0]:.4f}, {res['R_CI'][1]:.4f}]", suppress=False)

                                res = variance_partition_entropy_cues(dv_series=df_model['delegate_choice'], entropy_series=df_model['capabilities_entropy'], control_series_list=[df_model['o_prob']]+continuous_controls+categorical_controls)
                                log_output(f"Variance partitioning: Entropy unique R2={res['unique_entropy']:.4f} [{res['unique_entropy_CI'][0]:.4f}, {res['unique_entropy_CI'][1]:.4f}]", suppress=False)
                                log_output(f"Variance partitioning: Surface unique R2={res['unique_cues']:.4f} [{res['unique_cues_CI'][0]:.4f}, {res['unique_cues_CI'][1]:.4f}]", suppress=False)
                                log_output(f"Variance partitioning: Shared R2={res['shared']:.4f} [{res['shared_CI'][0]:.4f}, {res['shared_CI'][1]:.4f}]", suppress=False)
                                log_output(f"Variance partitioning: Unexplained R2={res['unexplained']:.4f} [{res['unexplained_CI'][0]:.4f}, {res['unexplained_CI'][1]:.4f}]", suppress=False)

                        except Exception as e_full:
                            log_output(f"                    Could not compute partial correlation on capent: {e_full}", suppress=False)
                        log_output(f"\n====================Partial correlation on decision: topprob====================")
                        try:
                            res = partial_correlation_on_decision(dv_series=1-df_model['delegate_choice'], iv_series=df_model['p_i_capability'], control_series_list=continuous_controls+categorical_controls)
                            log_output(res)
                        except Exception as e_full:
                            log_output(f"                    Could not compute partial correlation on topprob: {e_full}", suppress=False)

                        if 'o_prob' in df_model.columns and df_model['o_prob'].notna().any():
                            log_output(f"\n====================Regression of controls on oprob ====================")
                            try:
                                res = regression_std(dv_series=df_model['o_prob'], iv_series=None, control_series_list=continuous_controls+categorical_controls)
                                log_output(res['full_results'].summary())
                            except Exception as e_full:
                                log_output(f"                    Could not compute regression of controls on oprob: {e_full}")

                        log_output(f"\n====================Calibration metrics====================")
                        calib_dict = brier_ece(correctness_series=df_model['s_i_capability'], probability_series=df_model['p_i_capability'])
                        log_output(f"Brier Resolution (ranking): {calib_dict['resolution']:.4f} [{calib_dict['resolution_ci'][0]:.4f}, {calib_dict['resolution_ci'][1]:.4f}]", suppress=False)
                        log_output(f"Brier Reliability (reality): {calib_dict['reliability']:.4f} [{calib_dict['reliability_ci'][0]:.4f}, {calib_dict['reliability_ci'][1]:.4f}]", suppress=False)
                        log_output(f"Brier: {calib_dict['brier']:.4f} [{calib_dict['brier_ci'][0]:.4f}, {calib_dict['brier_ci'][1]:.4f}]", suppress=False)
                        log_output(f"ECE: {calib_dict['ece']:.4f} [{calib_dict['ece_ci'][0]:.4f}, {calib_dict['ece_ci'][1]:.4f}]", suppress=False)
                        log_output(f"df_model[p_i_capability] mean: {df_model['p_i_capability'].mean():.4f}, std: {df_model['p_i_capability'].std():.4f}", suppress=False)


                    if 'normalized_prob_entropy' in df_model.columns and df_model['normalized_prob_entropy'].notna().any():
                        # Model 4.5: normalized_prob_entropy in full model
                        final_model_terms_m45 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t)]
                        final_model_terms_m45.append('normalized_prob_entropy')
                        model_def_str_4_5 = 'delegate_choice ~ ' + ' + '.join(final_model_terms_m45)
                        log_output(f"\n                  Model 4.7: {model_def_str_4_5}")
                        try:
                            logit_m2 = smf.logit(model_def_str_4_5, data=df_model.dropna(subset=['normalized_prob_entropy', 'delegate_choice'])).fit(disp=0)
                            log_output(logit_m2.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 4.7: {e_full}")

                        # Model 4.6: normalized_prob_entropy in full model w/o s_i_capability
                        final_model_terms_m45 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t) and t != 's_i_capability']
                        final_model_terms_m45.append('normalized_prob_entropy')
                        model_def_str_4_5 = 'delegate_choice ~ ' + ' + '.join(final_model_terms_m45)
                        log_output(f"\n                  Model 4.8: {model_def_str_4_5}")
                        try:
                            logit_m2 = smf.logit(model_def_str_4_5, data=df_model.dropna(subset=['normalized_prob_entropy', 'delegate_choice'])).fit(disp=0)
                            log_output(logit_m2.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 4.8: {e_full}")

                        if 'o_prob' in df_model.columns and df_model['o_prob'].notna().any():
                            final_model_terms_m45.append('o_prob')
                            model_def_str_4_5 = 'delegate_choice ~ ' + ' + '.join(final_model_terms_m45)
                            if 'sp_prob' in df_model.columns and df_model['sp_prob'].notna().any():
                                final_model_terms_m45.append('sp_prob')
                                model_def_str_4_5 = 'delegate_choice ~ ' + ' + '.join(final_model_terms_m45)

                                #log_output(f"\n                  Answer Choice by Stated (Self and Other) vs GameEnt Model")
                                #res = compare_predictors_of_choice(df_model['sp_prob'], df_model['o_prob'], df_model['normalized_prob_entropy'], df_model['delegate_choice'])
                                #log_output(res)

                                log_output(f"\n====================Simplified Complete GameEnt Analysis====================")
                                continuous_controls = [df_model[t] for t in final_model_terms_m45 if t not in ['sp_prob', 'o_prob', 'normalized_prob_entropy'] and not (isinstance(t, str) and t.startswith('C('))]
                                categorical_controls = [df_model[t.replace('C(', '').replace(')', '')] for t in final_model_terms_m45 if (isinstance(t, str) and t.startswith('C('))]
                                #res, res_dict = compare_predictors_of_choice_simple(df_model['sp_prob'], df_model['o_prob'], df_model['normalized_prob_entropy'], df_model['delegate_choice'], continuous_controls, categorical_controls)
                                #log_output(res)
                                #res_dicts[model_name_part]['gameent'] = res_dict
                                log_output(f"\n==================== Entropy as Predictor of Decision Prob vs Stated Prob ====================")
                                try:
                                    results = compare_partial_correlations(predictor_series=df_model['capabilities_entropy'],
                                                                        outcome1_series=df_model['t_prob'],
                                                                        outcome2_series=1-df_model['sp_prob'],
                                                                        control_series_list=continuous_controls+categorical_controls+[df_model['o_prob']])
                                    log_output(f"Partial correlation (entropy → game), all controls: {results['partial_corr_outcome1']:.3f}", suppress=False)
                                    log_output(f"Partial correlation (entropy → stated), all controls: {results['partial_corr_outcome2']:.3f}", suppress=False)
                                    log_output(f"Decision Prob minus Stated Prob entropy correlation, all controls: {results['difference']:.3f} [{results['difference_ci'][0]:.3f}, {results['difference_ci'][1]:.3f}]", suppress=False)
                                    log_output(f"Steiger's test: z = {results['steiger_z']:.2f}, p = {results['p_value']:.3f}", suppress=False)

                                    results = compare_partial_correlations(predictor_series=df_model['capabilities_entropy'],
                                                                        outcome1_series=df_model['t_prob'],
                                                                        outcome2_series=1-df_model['sp_prob'])
                                    log_output(f"Partial correlation (entropy → game): {results['partial_corr_outcome1']:.3f}", suppress=False)
                                    log_output(f"Partial correlation (entropy → stated): {results['partial_corr_outcome2']:.3f}", suppress=False)
                                    log_output(f"Decision Prob minus Stated Prob entropy correlation, no controls: {results['difference']:.3f} [{results['difference_ci'][0]:.3f}, {results['difference_ci'][1]:.3f}]", suppress=False)
                                    log_output(f"Steiger's test: z = {results['steiger_z']:.2f}, p = {results['p_value']:.3f}", suppress=False)

                                    results = compare_partial_correlations(predictor_series=df_model['capabilities_entropy'],
                                                                        outcome1_series=1-df_model['sp_prob'],
                                                                        outcome2_series=1-df_model['o_prob'])
                                    log_output(f"Partial correlation (entropy → stated): {results['partial_corr_outcome1']:.3f}", suppress=False)
                                    log_output(f"Partial correlation (entropy → other): {results['partial_corr_outcome2']:.3f}", suppress=False)
                                    log_output(f"Stated Prob minus Other Prob entropy correlation, no controls: {results['difference']:.3f} [{results['difference_ci'][0]:.3f}, {results['difference_ci'][1]:.3f}]", suppress=False)
                                    log_output(f"Steiger's test: z = {results['steiger_z']:.2f}, p = {results['p_value']:.3f}", suppress=False)

                                    results = compare_surface_contamination(outcome1_series=df_model['t_prob'], outcome2_series=1-df_model['sp_prob'],control_series_list=continuous_controls+categorical_controls+[df_model['o_prob']])
                                    log_output(f"Influence of surface confounds on game decisions: R² = {results['r2_outcome1']:.3f} [{results['r2_outcome1_ci'][0]:.3f}, {results['r2_outcome1_ci'][1]:.3f}]", suppress=False)
                                    log_output(f"Influence of surface confounds on stated confidence: R² = {results['r2_outcome2']:.3f} [{results['r2_outcome2_ci'][0]:.3f}, {results['r2_outcome2_ci'][1]:.3f}]", suppress=False)
                                    log_output(f"Influence of surface confounds on game-stated: {results['r2_difference']:.3f} [{results['r2_difference_ci'][0]:.3f}, {results['r2_difference_ci'][1]:.3f}]", suppress=False)
                                except Exception as e_full:
                                    log_output(f"                    Could not compute entropy as predictor of decision prob vs stated prob: {e_full}", suppress=False)

                            log_output(f"\n====================Partial correlation on decision prob====================")
                            try:
                                res = partial_correlation_on_decision(dv_series=df_model['t_prob'], iv_series=df_model['capabilities_entropy'], control_series_list=continuous_controls+categorical_controls)
                                log_output(f"Partial correlation on decision prob with Capent, surface controls: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)
                                res = partial_correlation_on_decision(dv_series=df_model['t_prob'], iv_series=df_model['capabilities_entropy'], control_series_list=[df_model['o_prob']])
                                log_output(f"Partial correlation on decision prob with Capent, Stated Other control: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)
                                res = partial_correlation_on_decision(dv_series=df_model['t_prob'], iv_series=df_model['capabilities_entropy'], control_series_list=[df_model['o_prob']]+continuous_controls+categorical_controls)
                                log_output(f"Partial correlation on decision prob with Capent, all controls: {res['correlation']:.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]", suppress=False)
                            except Exception as e_full:
                                log_output(f"                    Could not compute partial correlation on decision prob: {e_full}", suppress=False)

                            results = regression_std(dv_series=df_model['t_prob'], iv_series=df_model['capabilities_entropy'], control_series_list=continuous_controls+categorical_controls)
                            res = results['full_results']
                            ci_lower, ci_upper = res.conf_int().loc['capabilities_entropy']
                            log_output(f"Linres on decision prob with Capent, surface controls: {res.params['capabilities_entropy']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], z={res.tvalues['capabilities_entropy']:.4f}, stderr={res.bse['capabilities_entropy']:.4f}",suppress=False)
                            results = regression_std(dv_series=df_model['t_prob'], iv_series=df_model['capabilities_entropy'], control_series_list=None)
                            res = results['full_results']
                            ci_lower, ci_upper = res.conf_int().loc['capabilities_entropy']
                            log_output(f"Linres on decision prob with Capent, no controls: {res.params['capabilities_entropy']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], z={res.tvalues['capabilities_entropy']:.4f}, stderr={res.bse['capabilities_entropy']:.4f}",suppress=False)
                            results = regression_std(dv_series=df_model['t_prob'], iv_series=df_model['capabilities_entropy'], control_series_list=[df_model['o_prob']])
                            res = results['full_results']
                            ci_lower, ci_upper = res.conf_int().loc['capabilities_entropy']
                            log_output(f"Linres on decision prob with Capent, Stated Other control: {res.params['capabilities_entropy']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], z={res.tvalues['capabilities_entropy']:.4f}, stderr={res.bse['capabilities_entropy']:.4f}",suppress=False)
                            results = regression_std(dv_series=df_model['t_prob'], iv_series=df_model['capabilities_entropy'], control_series_list=continuous_controls+categorical_controls+[df_model['o_prob']])
                            res = results['full_results']
                            ci_lower, ci_upper = res.conf_int().loc['capabilities_entropy']
                            log_output(f"Linres on decision prob with Capent, all controls: {res.params['capabilities_entropy']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], z={res.tvalues['capabilities_entropy']:.4f}, stderr={res.bse['capabilities_entropy']:.4f}",suppress=False)
                            z, z_ci_low, z_ci_high = (res.params['capabilities_entropy']/res.bse['capabilities_entropy'],) + tuple((res.conf_int().loc['capabilities_entropy'] / res.bse['capabilities_entropy']).values)
                            log_output(f"Linres on decision prob with Capent, all controls, standardized: {z:.4f} [{z_ci_low:.4f}, {z_ci_high:.4f}]", suppress=False)



                    # Model 5.5 (If judge_delegate was used in Model 5, do a model without it)
                    if 'judge_delegate' in final_model_terms_m5:
                        final_model_terms_m55 = [t for t in final_model_terms_m5 if t != 'judge_delegate' and t != 'teammate_judge_delegate']
                        model_def_str_5_5 = 'delegate_choice ~ ' + ' + '.join(final_model_terms_m55)
                        log_output(f"\n                  Model 5.5: {model_def_str_5_5}")
                        try:
                            logit_model5_5 = smf.logit(model_def_str_5_5, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model5_5.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 5.5: {e_full}")
                    
                    # Model 5.6 (If judge_delegate_combined exists, do a model with it)
                    if 'judge_delegate_combined' in df_model.columns:
                        final_model_terms_m56 = [t for t in final_model_terms_m5 if t != 'judge_delegate' and t != 'teammate_judge_delegate']
                        final_model_terms_m56.append('judge_delegate_combined')
                        model_def_str_5_6 = 'delegate_choice ~ ' + ' + '.join(final_model_terms_m56)
                        log_output(f"\n                  Model 5.6: {model_def_str_5_6}")
                        try:
                            logit_model5_6 = smf.logit(model_def_str_5_6, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model5_6.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 5.6: {e_full}")

                    # Model 7 (No s_i_capability, no interaction)
                    final_model_terms_m7 = [t for t in final_model_terms_m5 if t != 's_i_capability']
                    model_def_str_7 = 'delegate_choice ~ ' + ' + '.join(final_model_terms_m7)
                    log_output(f"\n                  Model 7: {model_def_str_7}", suppress=False)
                    try:
                        logit_model7 = smf.logit(model_def_str_7, data=df_model).fit(**fit_kwargs)
                        log_output(logit_model7.summary(), suppress=False)
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 7: {e_full}", suppress=False)

                    # Model XXX misused predictors
                    log_output(f"\n                  Model XXX: Analyzing misuse of predictors per model")
                    continuous_controls = [df_model[t] for t in final_model_terms_m7 if not (isinstance(t, str) and t.startswith('C('))]
                    categorical_controls = [df_model[t.replace('C(', '').replace(')', '')] for t in final_model_terms_m7 if (isinstance(t, str) and t.startswith('C('))]
                    try: 
                        res, models = analyze_wrong_way(df_model, continuous_controls, categorical_controls, alpha=0.05)
                        log_output(res.to_string(index=False))
                        misused_predictors = [row['predictor'] for _, row in res.iterrows() if row['misuse'] == True]
                        if misused_predictors:
                            log_output(f"Misused predictors: {misused_predictors}", suppress=False)
                        else:
                            log_output(f"No misused predictors found.", suppress=False)
                        misused_results.append(res)
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model XXX: {e_full}", suppress=False)

                    # Model 8 (judge_delegate only)
                    if 'judge_delegate' in df_model.columns:
                        model_def_str_8 = 'delegate_choice ~ judge_delegate'
                        log_output(f"\n                  Model 8: {model_def_str_8}")
                        try:
                            # Use original fit_kwargs which might include clustering by q_id
                            logit_model8 = smf.logit(model_def_str_8, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model8.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 8: {e_full}")

                    # Model 8.5 (teammate_judge_delegate only)
                    if 'teammate_judge_delegate' in df_model.columns:
                        model_def_str_8 = 'delegate_choice ~ teammate_judge_delegate'
                        log_output(f"\n                  Model 8.5: {model_def_str_8}")
                        try:
                            # Use original fit_kwargs which might include clustering by q_id
                            logit_model8 = smf.logit(model_def_str_8, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model8.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 8.5: {e_full}")

                    # Model 8.6 (judge_delegates only)
                    if 'teammate_judge_delegate' in df_model.columns and 'judge_delegate' in df_model.columns:
                        model_def_str_8 = 'delegate_choice ~ teammate_judge_delegate + judge_delegate'
                        log_output(f"\n                  Model 8.6: {model_def_str_8}")
                        try:
                            # Use original fit_kwargs which might include clustering by q_id
                            logit_model8 = smf.logit(model_def_str_8, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model8.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 8.6: {e_full}")

                    # Model 9 (judge_delegate but not other surface regessors)
                    if 'judge_delegate' in df_model.columns:
                        model_def_str_9 = 'delegate_choice ~ judge_delegate + s_i_capability + teammate_skill_ratio'
                        log_output(f"\n                  Model 9: {model_def_str_9}")
                        try:
                            # Use original fit_kwargs which might include clustering by q_id
                            logit_model9 = smf.logit(model_def_str_9, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model9.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 9: {e_full}")

                    # Model 10 (judge_delegate vs s_i_capability)
                    if 'judge_delegate' in df_model.columns:
                        model_def_str_10 = 'delegate_choice ~ judge_delegate + s_i_capability'
                        log_output(f"\n                  Model 10: {model_def_str_10}")
                        try:
                            # Use original fit_kwargs which might include clustering by q_id
                            logit_model10 = smf.logit(model_def_str_10, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model10.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 10: {e_full}")
                            
                    # Model 10.5 (teammate_judge_delegate vs s_i_capability)
                    if 'teammate_judge_delegate' in df_model.columns:
                        model_def_str_10 = 'delegate_choice ~ teammate_judge_delegate + s_i_capability'
                        log_output(f"\n                  Model 10.5: {model_def_str_10}")
                        try:
                            # Use original fit_kwargs which might include clustering by q_id
                            logit_model10 = smf.logit(model_def_str_10, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model10.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 10.5: {e_full}")
                            
                else:
                    log_output("\n                  Skipping Full Models due to insufficient data points (<=20).", print_to_console=True)

            except Exception as e:
                print(f"                  Error during logistic regression for {log_context_str}: {e}")
            
            print("-" * 40)


    qtype = "factual" if dataset in ['SimpleQA', 'SimpleMC'] else "reasoning"
    rtype = "mc" if dataset in ['SimpleMC', 'GPQA'] else "sa"
    with open(f"res_dicts_{qtype}_{rtype}_{game_type}.json", 'w') as f:
        json.dump(res_dicts, f, indent=2)

    try:
        summary = summarize_wrong_wayB(misused_results, alpha=0.05)
        log_output(f"\n\n{summary['conclusion']} (k={summary['n_wrong_way']}/{summary['n_candidates']}, expected={summary['expected_by_chance']:.1f}, p={summary['p_value']:.4f})", suppress=False)    
        from scipy.stats import binomtest
        p_value = binomtest(summary['n_wrong_way'], n=summary['n_candidates'], p=0.0, alternative='greater')
        log_output(f"Exact binomial test against 0 chance rate: {p_value}", suppress=False)
    except Exception as e:
        log_output(f"Could not summarize misuse results (method B): {e}", suppress=False)
    
    try: 
        potential_misuses = summarize_wrong_wayC(misused_results, alpha=0.05)
        n_potential = len(potential_misuses)
        if n_potential > 0:
            n_significant = potential_misuses['misuse_fdr'].sum()
            log_output(f"\nPotential misuses tested: {n_potential}; Significant after FDR correction: {n_significant}; Significant misused predictors: ", suppress=False)
            significant_misuses = potential_misuses[potential_misuses['misuse_fdr']]
            if len(significant_misuses) > 0:
                log_output(significant_misuses[['predictor', 'beta_correct', 'beta_delegate', 'p_one_sided_delegate_gt0', 'p_adjusted']], suppress=False)
            else:
                log_output("None survived FDR correction", suppress=False)
        else:
            log_output("\nNo potential misuses found (no predictors with positive baseline and positive delegation effects)", suppress=False)
    except Exception as e:
        log_output(f"Could not summarize misuse results (method C): {e}", suppress=False)