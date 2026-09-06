import json
import numpy as np
import pandas as pd
from pathlib import Path
from project_paths import RESULTS_ROOT, RXRX1_ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RXRX1_DIR  = RXRX1_ROOT
RESULT_DIR = RESULTS_ROOT / "rxrx1_analysis"
VIS_DIR    = RESULTS_ROOT / "visualisations"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Load metadata
    meta_path = RXRX1_DIR / "rxrx1" / "metadata.csv"
    if not meta_path.exists():
        meta_path = RXRX1_DIR / "metadata.csv"
    
    df = pd.read_csv(meta_path)
    print(f"Total images: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())

    # Analyse distribution
    print("\n===== DATASET STATISTICS =====")
    
    # Cell lines
    if "cell_type" in df.columns:
        cell_types = df["cell_type"].value_counts()
        print(f"\nCell types:\n{cell_types}")
    
    # Experiments/batches
    if "experiment" in df.columns:
        experiments = df["experiment"].value_counts()
        print(f"\nExperiments: {len(experiments)}")
        print(experiments.head(10))

    # Plates
    if "plate" in df.columns:
        plates = df["plate"].value_counts()
        print(f"\nPlates: {len(plates)}")

    # Sirna targets
    if "sirna" in df.columns:
        sirnas = df["sirna"].nunique()
        print(f"\nUnique siRNA targets: {sirnas}")

    # ── Stratified split ─────────────────────────────
    # Split by cell_type and experiment batch
    # Test set: hold out entire experiments not seen in training
    if "cell_type" in df.columns and "experiment" in df.columns:
        np.random.seed(42)
        
        # Get unique experiments per cell type
        exp_by_cell = df.groupby("cell_type")["experiment"].unique()
        
        train_exps, val_exps, test_exps = [], [], []
        for cell_type, exps in exp_by_cell.items():
            exps = list(exps)
            np.random.shuffle(exps)
            n = len(exps)
            train_exps.extend(exps[:int(n*0.7)])
            val_exps.extend(exps[int(n*0.7):int(n*0.85)])
            test_exps.extend(exps[int(n*0.85):])

        df["split"] = "train"
        df.loc[df["experiment"].isin(val_exps),  "split"] = "val"
        df.loc[df["experiment"].isin(test_exps), "split"] = "test"

        split_summary = df.groupby(["split", "cell_type"]).size().unstack(fill_value=0)
        print(f"\n===== STRATIFIED SPLIT =====")
        print(split_summary)

        df.to_csv(RESULT_DIR / "rxrx1_stratified_split.csv", index=False)

        # Save split info
        split_info = {
            "train_experiments": train_exps,
            "val_experiments":   val_exps,
            "test_experiments":  test_exps,
            "n_train": int((df["split"]=="train").sum()),
            "n_val":   int((df["split"]=="val").sum()),
            "n_test":  int((df["split"]=="test").sum()),
        }
        with open(RESULT_DIR / "split_info.json", "w") as f:
            json.dump(split_info, f, indent=2)

    # ── Visualisations ───────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cell type distribution
    if "cell_type" in df.columns:
        cell_counts = df["cell_type"].value_counts()
        axes[0,0].bar(cell_counts.index, cell_counts.values,
                      color=["#4C72B0","#55A868","#C44E52","#8172B2"],
                      edgecolor='black')
        axes[0,0].set_title("Images per Cell Type", fontsize=13)
        axes[0,0].set_ylabel("Number of Images")
        for i, v in enumerate(cell_counts.values):
            axes[0,0].text(i, v+100, str(v), ha='center', fontsize=10)

    # 2. Split distribution by cell type
    if "cell_type" in df.columns:
        split_ct = df.groupby(["cell_type","split"]).size().unstack(fill_value=0)
        split_ct.plot(kind='bar', ax=axes[0,1],
                      color=["#C44E52","#55A868","#4C72B0"],
                      edgecolor='black')
        axes[0,1].set_title("Train/Val/Test Split by Cell Type", fontsize=13)
        axes[0,1].set_ylabel("Number of Images")
        axes[0,1].legend(title="Split")
        axes[0,1].tick_params(axis='x', rotation=30)

    # 3. Experiments per cell type
    if "cell_type" in df.columns and "experiment" in df.columns:
        exp_counts = df.groupby("cell_type")["experiment"].nunique()
        axes[1,0].bar(exp_counts.index, exp_counts.values,
                      color="#8172B2", edgecolor='black')
        axes[1,0].set_title("Experiments per Cell Type", fontsize=13)
        axes[1,0].set_ylabel("Number of Experiments")

    # 4. Overall split pie
    if "split" in df.columns:
        split_counts = df["split"].value_counts()
        axes[1,1].pie(split_counts.values,
                      labels=[f"{k}\n({v})" for k,v in split_counts.items()],
                      colors=["#4C72B0","#55A868","#C44E52"],
                      autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title("Overall Train/Val/Test Split", fontsize=13)

    plt.suptitle("RxRx1 Dataset Analysis & Stratified Split", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "rxrx1_dataset_analysis.png", dpi=150, bbox_inches='tight')
    print("\nSaved rxrx1_dataset_analysis.png")
    print("All done.")

if __name__ == "__main__":
    main()
