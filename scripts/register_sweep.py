import yaml
import wandb

def register_sweep():
    with open("sweep.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)

    # The project name should be consistent with the one in the configs.
    project_name = "differentiable-svm-tree"

    sweep_id = wandb.sweep(
        sweep_config,
        project=project_name,
    )
    print(f"Sweep registered with ID: {sweep_id}")

if __name__ == "__main__":
    register_sweep()