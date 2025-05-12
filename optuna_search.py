# optuna_search.py
import io, copy, yaml, optuna
import os, shutil

from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from train_liif import run_once
"""
Optuna Hyperparameter Optimization for LIIF Super-Resolution Model

This script performs automated hyperparameter optimization (HPO) for the LIIF
(super-resolution) model using Optuna. It leverages a base YAML config file 
and dynamically overrides key hyperparameters such as learning rate, training 
epochs, repeat count, and learning rate decay schedule.

The optimization loop:
- Samples hyperparameters using a TPE sampler.
- Applies early stopping via Hyperband pruning.
- Runs one training session per trial via `run_once`.
- Tracks and prints validation L1 loss per trial.
- Best parameters and best validation loss are printed at the end.

"""
# ============ Global Settings ============ #
BASE_YAML  = 'configs/single-edsr.yaml'  # Path to base YAML config
N_TRIALS   = 40                          # Number of Optuna trials
TIMEOUT_S  = 3600                        # Max total run time in seconds (e.g., 1 hour)
# ---------------------------------- #
def objective(trial):
    trial_id  = f"optuna_t{trial.number}"
    save_path = f"./save/{trial_id}"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    with io.open(BASE_YAML, 'r', encoding='utf-8-sig') as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)

    # ============  Settings ============ #

    lr     = trial.suggest_float('lr', 1e-5, 3e-4, log=True)
    repeat = trial.suggest_int('repeat', 50, 200)
    epochs = trial.suggest_int('epoch_max', 20, 100)

    frac1 = trial.suggest_float('decay1_frac', 0.3, 0.6)
    frac2 = trial.suggest_float('decay2_frac', 0.7, 0.9)
    gamma = trial.suggest_float('gamma', 0.4, 0.8)

    # ---------------------------------- #

    m1 = max(1, int(epochs * frac1))
    m2 = max(m1 + 1, int(epochs * frac2))

    cfg['optimizer']['args']['lr'] = lr
    cfg['train_dataset']['dataset']['args']['repeat'] = repeat
    cfg['epoch_max'] = epochs
    cfg['multi_step_lr'] = {'milestones': [m1, m2], 'gamma': gamma}

    val_loss = run_once(cfg, save_path=save_path, trial=trial)

    trial.set_user_attr('val_loss', val_loss)
    print(f"[Trial {trial.number:02d}] "
          f"lr={lr:.1e}, repeat={repeat}, epochs={epochs}, "
          f"ms=[{m1},{m2}], γ={gamma:.2f} → val_L1={val_loss:.4f}")
    return val_loss


if __name__ == '__main__':
    sampler = TPESampler(multivariate=True, group=True, seed=42)

    # Use Hyperband pruner (based on epoch count)
    # max_resource will be set dynamically per trial (equal to sampled epochs)
    pruner = HyperbandPruner(min_resource=5,  #  Start pruning after at least 5 epochs
                             reduction_factor=3)

    study = optuna.create_study(direction='minimize',
                                sampler=sampler,
                                pruner=pruner)

    print(f"=== Optuna HPO: {N_TRIALS} trials, timeout={TIMEOUT_S//60} min ===")
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_S)
    print("\n========= HPO FINISH =========")
    print("Best params :", study.best_params)
    print("Best val_L1 :", study.best_value)