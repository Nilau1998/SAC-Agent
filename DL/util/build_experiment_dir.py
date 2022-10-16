import os
import time

def build_experiment_dir(experiment_name="experiment"):
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)

    # Build new experiment directory
    timestamp = time.strftime("_%m-%d_%H-%M-%S")
    experiment_name = experiment_name
    experiment_dir = os.path.join(
            experiments_dir,
            experiment_name + timestamp
        )
    os.makedirs(os.path.join(experiment_dir, "plots"))
    os.makedirs(os.path.join(experiment_dir, "checkpoints"))
    print(f"Created: {experiment_dir}")
    return experiment_dir