import json
import subprocess
import sys
from pathlib import Path


REQUIRED_FILES = [
    "REPORT.docx",
    "LEARNING_CURVES.docx",
    "REPORT.md",
    "ROLES.md",
    "AI_USE.md",
    "README.md",
    "reproduce.sh",
    "requirements.txt",
    "logs/final_results_summary.json",
    "logs/bc_ablation_summary.json",
    "logs/bc_3seed_summary.json",
    "weights/bc_policy_t055.pt",
    "weights/bc_policy_t055_seed1.pt",
    "weights/bc_policy_t055_seed2.pt",
]


def run_command(command):
    print(f"\nRunning: {' '.join(command)}")
    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"Command failed: {' '.join(command)}")
        sys.exit(result.returncode)


def check_required_files():
    print("\nChecking required files...")

    missing = []

    for file_path in REQUIRED_FILES:
        path = Path(file_path)
        if path.exists():
            print(f"[OK] {file_path}")
        else:
            print(f"[MISSING] {file_path}")
            missing.append(file_path)

    if missing:
        print("\nSome required files are missing.")
        sys.exit(1)

    print("\nAll required files exist.")


def print_final_summary():
    summary_path = Path("logs/final_results_summary.json")

    if not summary_path.exists():
        print("\nlogs/final_results_summary.json not found.")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print("\nFinal Results Summary")
    print("-" * 50)

    results = summary.get("results", {})

    for method, values in results.items():
        if "cost_per_order" in values:
            print(f"{method}: cost_per_order = {values['cost_per_order']}")
        elif "mean_cost_per_order" in values:
            print(
                f"{method}: mean_cost_per_order = "
                f"{values['mean_cost_per_order']} ± {values.get('std_cost_per_order')}"
            )

    key_findings = summary.get("key_findings", {})
    if key_findings:
        print("\nKey Findings")
        print("-" * 50)
        for key, value in key_findings.items():
            print(f"{key}: {value}")


def main():
    print("IE306 Drone Dispatch Reproducibility Check")
    print("=" * 50)

    run_command([sys.executable, "-m", "pytest", "tests", "-q"])
    check_required_files()
    print_final_summary()

    print("\nReproducibility check completed successfully.")


if __name__ == "__main__":
    main()