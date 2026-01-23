import subprocess
from pathlib import Path

def run(input_files, output_files, params, **kwargs):
    """
    Python wrapper to run DeepCyTOF within the OmniBenchmark module system.
    """

    # 1. Extract input paths from the OmniBenchmark input dictionary
    train_matrix = input_files["data.train_matrix"]
    train_labels = input_files["data.train_labels"]
    test_matrix = input_files["data.test_matrix"]

    # 2. Extract output path and ensure the directory exists
    # Note: Ensure the key "analysis.prediction.deepcytoftool" matches your Snakefile/config
    pred_path = Path(output_files["analysis.prediction.deepcytoftool"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Get the path to the internal runner script
    # This points to: ob-pipeline-deepcytof/deepcytof_pipeline/run_deepcytof.py
    run_script = Path(__file__).resolve().parents[1] / "deepcytof_pipeline" / "run_deepcytof.py"

    # 4. Construct the command
    # We use the flags defined in our run_deepcytof.py script
    cmd = [
        "python",
        str(run_script),
        "--train_x", str(train_matrix),
        "--train_y", str(train_labels),
        "--test_x", str(test_matrix),
        "--output_file", str(pred_path),
        "--dataset_name", kwargs.get("name", "deepcytof_run")
    ]

    print("🚀 OmniBenchmark Module executing DeepCyTOF...")
    print("Running:", " ".join(cmd))
    
    # 5. Execute
    subprocess.check_call(cmd)