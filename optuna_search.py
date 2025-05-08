# optuna_search.py
import io, copy, yaml, optuna
import os, shutil

from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from train_liif import run_once

# ============ 全局设置 ============ #
BASE_YAML  = 'configs/single-edsr.yaml'  # 基础配置
N_TRIALS   = 40                          # 搜索次数
TIMEOUT_S  = 3600                        # 最长 2 h

# ---------------------------------- #
def objective(trial):
    trial_id  = f"optuna_t{trial.number}"
    save_path = f"./save/{trial_id}"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    # 1) 读取并深拷贝 yaml
    with io.open(BASE_YAML, 'r', encoding='utf-8-sig') as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)

    # 2) 采样核心超参
    lr     = trial.suggest_float('lr', 1e-5, 3e-4, log=True)
    repeat = trial.suggest_int('repeat', 50, 200)
    epochs = trial.suggest_int('epoch_max', 20, 100)

    # 3) 采样 LR 调度
    frac1 = trial.suggest_float('decay1_frac', 0.3, 0.6)
    frac2 = trial.suggest_float('decay2_frac', 0.7, 0.9)
    gamma = trial.suggest_float('gamma', 0.4, 0.8)
    m1 = max(1, int(epochs * frac1))
    m2 = max(m1 + 1, int(epochs * frac2))

    # 4) 写回 cfg
    cfg['optimizer']['args']['lr'] = lr
    cfg['train_dataset']['dataset']['args']['repeat'] = repeat
    cfg['epoch_max'] = epochs
    cfg['multi_step_lr'] = {'milestones': [m1, m2], 'gamma': gamma}

    # 5) 运行一次训练（run_once 已含剪枝回调）
    val_loss = run_once(cfg, save_path=save_path, trial=trial)

    # 6) 记录 & 打印
    trial.set_user_attr('val_loss', val_loss)
    print(f"[Trial {trial.number:02d}] "
          f"lr={lr:.1e}, repeat={repeat}, epochs={epochs}, "
          f"ms=[{m1},{m2}], γ={gamma:.2f} → val_L1={val_loss:.4f}")
    return val_loss


if __name__ == '__main__':
    # 多变量 TPE 采样器
    sampler = TPESampler(multivariate=True, group=True, seed=42)

    # Hyperband 剪枝器（基于 epoch 数）
    # 这里 max_resource 在每个 trial 内自动设置：等于采样到的 epochs
    pruner = HyperbandPruner(min_resource=5,  # 至少跑 5 轮后才比较
                             reduction_factor=3)

    study = optuna.create_study(direction='minimize',
                                sampler=sampler,
                                pruner=pruner)

    print(f"=== Optuna HPO: {N_TRIALS} trials, timeout={TIMEOUT_S//60} min ===")
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_S)
    print("\n========= HPO FINISH =========")
    print("Best params :", study.best_params)
    print("Best val_L1 :", study.best_value)