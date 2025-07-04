import optuna
import subprocess
import uuid
import parameters
import os
import re

def modify_parameters(trial):
    params = parameters.get_params()
    params['optimizer'] = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
    params['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.03, 0.06)
    params['batch_size'] = trial.suggest_categorical('batch_size', [64, 128])
    params['nb_cnn2d_filt'] = trial.suggest_categorical('nb_cnn2d_filt', [64, 128])
    params['nb_rnn_layers'] = trial.suggest_int('nb_rnn_layers', 1, 2)
    params['nb_self_attn_layers'] = trial.suggest_int('nb_self_attn_layers', 1, 3)
    params['weight_decay'] = trial.suggest_categorical('weight_decay', [0.01, 0.02, 0.05])
    params['rnn_size'] = 128
    params['nb_heads'] = trial.suggest_categorical('nb_heads', [4, 8])
    params['nb_fnn_layers'] = 1
    params['fnn_size'] = 128
    params['t_pool_size'] = [5, 1, 1]
    params['f_pool_size'] = [4, 4, 2]
    params['nb_epochs'] = 250
    params['patience'] = 50
    params['quick_test'] = False

    return params


def write_temp_parameter_override(params, file_path='temp_params.py'):
    with open(file_path, 'w') as f:
        f.write("def get_params():\n")
        f.write("    return {\n")
        for key, val in params.items():
            if isinstance(val, str):
                val_repr = f"r'{val}'" if "\\" in val else f"'{val}'"
            elif isinstance(val, float) and val == float('inf'):
                val_repr = "float('inf')"
            else:
                val_repr = val
            f.write(f"        '{key}': {val_repr},\n")
        f.write("    }\n")


'''def parse_seld_score(output_text):
    match = re.search(r'SELD_SCORE:\s*([0-9.]+)', output_text)
    if match:
        return float(match.group(1))
    return None'''

def parse_seld_score(output_text):
    matches = re.findall(r'SELD_SCORE:\s*([0-9.]+)', output_text)
    if matches:
        return float(matches[-1])
    return None


def objective(trial):
    job_id = str(uuid.uuid4())[:8]
    temp_param_file = "parameters_temp.py"
    
    # Replaces perameters.py
    params = modify_parameters(trial)
    write_temp_parameter_override(params, temp_param_file)
    os.replace(temp_param_file, os.path.join('SELD', 'parameters.py'))
    # Start training
    process = subprocess.Popen(
        ['python', os.path.join('SELD', 'train_seldnet.py'), job_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    output_lines = []
    print("=== OUTPUT (live) ===")
    for line in iter(process.stdout.readline, b''):
        decoded_line = line.decode(errors="ignore").rstrip()
        print(decoded_line)
        output_lines.append(decoded_line)
    print("==============")
    output_text = "\n".join(output_lines)

    seld_score = parse_seld_score(output_text)

    print(f"[Trial {job_id}] SELD score: {seld_score}")
    return seld_score if seld_score is not None else float('inf')


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials = 30)

    try:
        import joblib
        joblib.dump(study, "study.pkl")
    except ImportError:
        print("Joblib not installed - results will not be saved")
    
    print("Best parameters:")
    print(study.best_trial.params)