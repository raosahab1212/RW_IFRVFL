import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# import LightGBM as lgb
import lightgbm as lgb
import catboost as cb
from TRKM import TRKMClassifier
from GBTSVM import GBTSVMClassifier
from IFRVFL import IFRVFLC
from NFRVFL import NeuroFuzzyRVFLClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from ANFIS import HybridAlgorithmClassifier
from RWIFRVFL import RWIFRVFL
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

pd.set_option('display.max_columns', None)

def calculate_metrics(y_true, y_pred, y_score=None):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted',zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # ---------------------------------------------------------
    # ROC-AUC
    # ---------------------------------------------------------

    try:

        if y_score is not None:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                if np.ndim(y_score) == 2:
                    y_score_auc = y_score[:, 1]
                else:
                    y_score_auc = y_score
                roc_auc = roc_auc_score(y_true, y_score_auc)

            else:

                # Multiclass
                roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='weighted')

        else:
            roc_auc = np.nan

    except (ValueError, IndexError):
        roc_auc = np.nan

    return (accuracy, precision, recall, f1, roc_auc)

def get_model_score(model, X):
    # ---------------------------------------------------------
    # Preferred: predict_proba
    # ---------------------------------------------------------
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba

    # ---------------------------------------------------------
    # Second choice: decision_function
    # ---------------------------------------------------------
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)

    # ---------------------------------------------------------
    # Last fallback
    # ---------------------------------------------------------
    else:
        return model.predict(X)

def create_standard_models():

    models = {
        # =====================================================
        # Existing models
        # =====================================================
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=100, verbose=False, random_seed=42),
        "NB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")
    }
    return models

import time
import numpy as np
import pandas as pd




def run_10fold_cv(
    X,
    y,
    dataset_name="Dataset",
    # =========================================================
    # ANFIS parameters
    # =========================================================
    L=3,
    eta=0.1,
    num_iterations=100,
    num_cluster=3,
    fuzziness_index=2,
    a=1.0,
    epsilon=1e-5,
    lambda_reg=1e-3,
    # =========================================================
    # GBTSVM parameters
    # =========================================================
    d1=1.0,
    d2=1.0,
    purity=0.95,
    min_sample=5,
    min_ball_size=2,
    eps1=0.01,
    eps2=0.01,

    # =========================================================
    # TRKM parameters
    # =========================================================
    eta1=0.1,
    eta2=0.1,
    lambd1=1e-3,
    lambd2=1e-3,
    mew=1.0
):
    X = np.asarray(X)
    y = np.asarray(y)

    # =========================================================
    # 10-Fold Stratified Cross-Validation
    # =========================================================
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # =========================================================
    # Model names
    # =========================================================

    model_names = [
        "SVM",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "NB",
        "KNN",
        "NFRVFL",
        "ANFIS",
        "IFRVFL",
        "RWIFRVFL",
        "GBTSVM",
        "TRKM"
    ]

    # =========================================================
    # Standard ML models
    # =========================================================

    models = create_standard_models()
    # =========================================================
    # Store results
    # =========================================================

    fold_results = {name: [] for name in model_names}
    roc_data = {name: {"y_true": [], "y_score": []} for name in model_names}

    # =========================================================
    # 10-Fold CV
    # =========================================================

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print("\n" + "=" * 70)
        print(f"{dataset_name} - Fold {fold}/10")
        print("=" * 70)

        # -----------------------------------------------------
        # Train/Test split
        # -----------------------------------------------------
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # =====================================================
        # Standardization
        # =====================================================

        # IMPORTANT:
        # Fit scaler ONLY on training data.

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # =====================================================
        # 1-5. Standard ML Models
        # =====================================================

        for name, model in models.items():
            print(f"Running {name}...")
            start_time = time.time()
            # -------------------------------------------------
            # Training
            # -------------------------------------------------
            model.fit(X_train_scaled, y_train)

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------
            y_pred = model.predict(X_test_scaled)

            # -------------------------------------------------
            # Execution time
            # -------------------------------------------------

            execution_time = (time.time() - start_time)

            # -------------------------------------------------
            # ROC-AUC score
            # -------------------------------------------------
            y_score = get_model_score(model, X_test_scaled)

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------
            metrics = calculate_metrics(y_test, y_pred, y_score)
            fold_results[name].append([fold, *metrics, execution_time])

            # -------------------------------------------------
            # Store ROC information
            # -------------------------------------------------

            roc_data[name]["y_true"].extend(y_test.tolist())
            if np.ndim(y_score) == 1:
                roc_data[name]["y_score"].extend(y_score.tolist())

            elif y_score.shape[1] == 2:
                roc_data[name]["y_score"].extend(y_score[:, 1].tolist())

            else:
                roc_data[name]["y_score"].extend(y_score.tolist())

        # =====================================================
        # 6. NFRVFL
        # =====================================================

        print("Running NFRVFL...")
        start_time = time.time()
        nfrvfl = NeuroFuzzyRVFLClassifier(n_hidden=100, random_state=42)
        nfrvfl.fit(X_train_scaled, y_train)
        y_pred = nfrvfl.predict(X_test_scaled)
        y_score = nfrvfl.predict_proba(X_test_scaled)
        execution_time = (time.time() - start_time)
        metrics = calculate_metrics(y_test, y_pred, y_score)

        fold_results["NFRVFL"].append([fold, *metrics, execution_time])
        roc_data["NFRVFL"]["y_true"].extend(y_test.tolist())

        if y_score.shape[1] == 2:
            roc_data["NFRVFL"]["y_score"].extend(y_score[:, 1].tolist())

        else:
            roc_data["NFRVFL"]["y_score"].extend(y_score.tolist())

        # =====================================================
        # 7. ANFIS
        # =====================================================

        print("Running ANFIS...")

        start_time = time.time()

        model_anfis = HybridAlgorithmClassifier(
            X_train_scaled,
            np.asarray(y_train),
            L,
            eta,
            num_iterations,
            num_cluster,
            fuzziness_index,
            a,
            epsilon,
            lambda_reg,
            lambda1=1e-2,
            lambda2=1e-1
        )

        # -----------------------------------------------------
        # Training
        # -----------------------------------------------------

        model_anfis.forward()

        # -----------------------------------------------------
        # Prediction
        # -----------------------------------------------------
        y_pred_anfis = model_anfis.predict(X_test_scaled)
        execution_time = (time.time() - start_time)
        y_pred_anfis = np.asarray(y_pred_anfis).ravel()

        # -----------------------------------------------------
        # ANFIS probabilities
        # -----------------------------------------------------

        if hasattr(model_anfis, "predict_proba"):
            y_score_anfis = (model_anfis.predict_proba(X_test_scaled))

        else:
            y_score_anfis = y_pred_anfis

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        metrics = calculate_metrics(y_test, y_pred_anfis, y_score_anfis)
        fold_results["ANFIS"].append([fold, *metrics, execution_time])
        roc_data["ANFIS"]["y_true"].extend(y_test.tolist())

        if np.ndim(y_score_anfis) == 1:
            roc_data["ANFIS"]["y_score"].extend(y_score_anfis.tolist())

        elif y_score_anfis.shape[1] == 2:
            roc_data["ANFIS"]["y_score"].extend(y_score_anfis[:, 1].tolist())

        else:
            roc_data["ANFIS"]["y_score"].extend(y_score_anfis.tolist())

        # =====================================================
        # 8. IFRVFL
        # =====================================================

        print("Running IFRVFL...")
        start_time = time.time()
        model_ifrvflc = IFRVFLC(
            n_hidden=100,
            lam=1e-3,
            k=5,
            sigma=None,
            sigma_scale=1.0,
            random_state=42,
            normalize=True
        )

        model_ifrvflc.fit(X_train_scaled, y_train)
        y_pred_ifrvflc = model_ifrvflc.predict(X_test_scaled)
        execution_time = (time.time() - start_time)
        y_pred_ifrvflc = np.asarray(y_pred_ifrvflc).ravel()

        # -----------------------------------------------------
        # Score
        # -----------------------------------------------------
        if hasattr(model_ifrvflc, "predict_proba"):
            y_score_ifrvflc = (model_ifrvflc.predict_proba(X_test_scaled))

        elif hasattr(model_ifrvflc, "decision_function"):
            y_score_ifrvflc = (model_ifrvflc.decision_function(X_test_scaled))

        else:
            y_score_ifrvflc = y_pred_ifrvflc

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        metrics = calculate_metrics(y_test, y_pred_ifrvflc, y_score_ifrvflc)
        fold_results["IFRVFL"].append([fold, *metrics, execution_time])
        roc_data["IFRVFL"]["y_true"].extend(y_test.tolist())

        if np.ndim(y_score_ifrvflc) == 1:
            roc_data["IFRVFL"]["y_score"].extend(y_score_ifrvflc.tolist())

        elif y_score_ifrvflc.shape[1] == 2:
            roc_data["IFRVFL"]["y_score"].extend(y_score_ifrvflc[:, 1].tolist())

        else:
            roc_data["IFRVFL"]["y_score"].extend(y_score_ifrvflc.tolist())

        # =====================================================
        # 9. RWIFRVFL
        # =====================================================

        print("Running RWIFRVFL...")
        start_time = time.time()
        model_rwifrvfl = RWIFRVFL(k = 3, lam = 0.0001, n_hidden = 50, sigma_scale = 0.5)
        model_rwifrvfl.fit(X_train_scaled, y_train)
        y_pred_rwifrvfl = model_rwifrvfl.predict(X_test_scaled)
        execution_time = (time.time() - start_time)
        y_pred_rwifrvfl = np.asarray(y_pred_rwifrvfl).ravel()

        # -----------------------------------------------------
        # Score
        # -----------------------------------------------------

        if hasattr(model_rwifrvfl, "predict_proba"):
            y_score_rwifrvfl = (model_rwifrvfl.predict_proba(X_test_scaled))

        elif hasattr(model_rwifrvfl, "decision_function"):
            y_score_rwifrvfl = (model_rwifrvfl.decision_function(X_test_scaled))

        else:
            y_score_rwifrvfl = y_pred_rwifrvfl

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        metrics = calculate_metrics(y_test, y_pred_rwifrvfl, y_score_rwifrvfl)
        fold_results["RWIFRVFL"].append([fold, *metrics, execution_time])
        roc_data["RWIFRVFL"]["y_true"].extend(y_test.tolist())

        if np.ndim(y_score_rwifrvfl) == 1:
            roc_data["RWIFRVFL"]["y_score"].extend(y_score_rwifrvfl.tolist())

        elif y_score_rwifrvfl.shape[1] == 2:
            roc_data["RWIFRVFL"]["y_score"].extend(y_score_rwifrvfl[:, 1].tolist())

        else:
            roc_data["RWIFRVFL"]["y_score"].extend(y_score_rwifrvfl.tolist())

        # =====================================================
        # 10. GBTSVM
        # =====================================================

        print("Running GBTSVM...")
        start_time = time.time()

        # -----------------------------------------------------
        # Create GBTSVM model
        # -----------------------------------------------------

        model_gbtsvm = GBTSVMClassifier(
            d1=d1,
            d2=d2,
            purity=purity,
            min_sample=min_sample,
            min_ball_size=min_ball_size,
            eps1=eps1,
            eps2=eps2
        )

        # -----------------------------------------------------
        # Train
        # -----------------------------------------------------

        model_gbtsvm.fit(X_train_scaled, y_train)

        # -----------------------------------------------------
        # Prediction
        # -----------------------------------------------------

        y_pred_gbtsvm = model_gbtsvm.predict(X_test_scaled)

        # -----------------------------------------------------
        # Decision score
        # -----------------------------------------------------

        y_score_gbtsvm = (model_gbtsvm.decision_function(X_test_scaled))
        execution_time = (time.time() - start_time)
        y_pred_gbtsvm = np.asarray(y_pred_gbtsvm).ravel()

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        metrics = calculate_metrics(y_test, y_pred_gbtsvm, y_score_gbtsvm)
        fold_results["GBTSVM"].append([fold, *metrics, execution_time])

        # -----------------------------------------------------
        # ROC information
        # -----------------------------------------------------
        roc_data["GBTSVM"]["y_true"].extend(y_test.tolist())

        if np.ndim(y_score_gbtsvm) == 1:
            roc_data["GBTSVM"]["y_score"].extend(y_score_gbtsvm.tolist())

        elif y_score_gbtsvm.shape[1] == 2:
            roc_data["GBTSVM"]["y_score"].extend(y_score_gbtsvm[:, 1].tolist())

        else:
            roc_data["GBTSVM"]["y_score"].extend(y_score_gbtsvm.tolist())

        # =====================================================
        # 11. TRKM
        # =====================================================

        print("Running TRKM...")
        start_time = time.time()

        # -----------------------------------------------------
        # Create TRKM model
        # -----------------------------------------------------

        model_trkm = TRKMClassifier(
            eta1=eta1,
            eta2=eta2,
            lambd1=lambd1,
            lambd2=lambd2,
            mew=mew
        )

        # -----------------------------------------------------
        # Train
        # -----------------------------------------------------

        model_trkm.fit(X_train_scaled, y_train)

        # -----------------------------------------------------
        # Prediction
        # -----------------------------------------------------

        y_pred_trkm = model_trkm.predict(X_test_scaled)

        # -----------------------------------------------------
        # Decision score
        # -----------------------------------------------------

        y_score_trkm = (model_trkm.decision_function(X_test_scaled))
        execution_time = (time.time() - start_time)
        y_pred_trkm = np.asarray(y_pred_trkm).ravel()

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        metrics = calculate_metrics(y_test, y_pred_trkm, y_score_trkm)
        fold_results["TRKM"].append([fold, *metrics, execution_time])

        # -----------------------------------------------------
        # ROC information
        # -----------------------------------------------------

        roc_data["TRKM"]["y_true"].extend(y_test.tolist())

        if np.ndim(y_score_trkm) == 1:
            roc_data["TRKM"]["y_score"].extend(y_score_trkm.tolist())

        elif y_score_trkm.shape[1] == 2:
            roc_data["TRKM"]["y_score"].extend(y_score_trkm[:, 1].tolist())

        else:
            roc_data["TRKM"]["y_score"].extend(y_score_trkm.tolist())

    # =========================================================
    # Convert Fold Results to DataFrames
    # =========================================================
    fold_dfs = {}
    for name in model_names:
        fold_dfs[name] = pd.DataFrame(
            fold_results[name],
            columns=[
                "Fold",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
                "Execution Time"
            ]
        )

    # =========================================================
    # Summary: Mean ± SD
    # =========================================================
    summary_rows = []
    for name in model_names:
        df = fold_dfs[name]
        summary_rows.append(
            {
                "Model": name,
                "Accuracy Mean": df["Accuracy"].mean(),
                "Accuracy SD": df["Accuracy"].std(),
                "Precision Mean": df["Precision"].mean(),
                "Precision SD": df["Precision"].std(),
                "Recall Mean": df["Recall"].mean(),
                "Recall SD": df["Recall"].std(),
                "F1 Mean": df["F1"].mean(),
                "F1 SD": df["F1"].std(),
                "ROC-AUC Mean": df["ROC-AUC"].mean(),
                "ROC-AUC SD": df["ROC-AUC"].std(),
                "Execution Time Mean": df["Execution Time"].mean(),
                "Execution Time SD": df["Execution Time"].std(),
                "Total Execution Time": df["Execution Time"].sum()
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    # =========================================================
    # Print Summary
    # =========================================================
    print("\n")
    print("=" * 110)
    print(f"10-FOLD CROSS-VALIDATION RESULTS - {dataset_name}")
    print("=" * 110)
    print(summary_df.round(4).to_string(index=False))

    # =========================================================
    # Return
    # =========================================================
    return (fold_dfs, summary_df, roc_data)


def plot_roc_curves(roc_data, dataset_name="Dataset"):
    plt.figure(figsize=(10, 8))
    for model_name, data in roc_data.items():
        y_true = np.asarray(data["y_true"])
        y_score = np.asarray(data["y_score"])

        # =====================================================
        # Binary ROC only
        # =====================================================
        if y_score.ndim != 1:
            print(f"Skipping {model_name}: " "non-binary score format.")
            continue

        # Flatten
        y_true = y_true.ravel()
        y_score = y_score.ravel()

        # Check sample count
        if len(y_true) != len(y_score):
            print(f"Skipping {model_name}: " f"{len(y_true)} y_true vs " f"{len(y_score)} y_score.")
            continue

        # Check binary labels
        if len(np.unique(y_true)) != 2:
            print(f"Skipping {model_name}: " "test set does not contain two classes.")
            continue

        # =====================================================
        # Remove invalid values
        # =====================================================

        valid = (np.isfinite(y_true) & np.isfinite(y_score))
        y_true_valid = y_true[valid]
        y_score_valid = y_score[valid]

        # =====================================================
        # ROC Curve
        # =====================================================

        try:
            fpr, tpr, thresholds = roc_curve(y_true_valid, y_score_valid)
            auc_value = roc_auc_score(y_true_valid, y_score_valid)
        except ValueError as e:
            print(f"Skipping {model_name}: {e}")
            continue

        # =====================================================
        # Plot ROC
        # =====================================================

        plt.plot(fpr, tpr, linewidth=2)
            # label=(
            #     f"{model_name} "
            #     f"(AUC = {auc_value:.3f})"
            # )
        # )

    # =========================================================
    # Random Classifier
    # =========================================================

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)

    # =========================================================
    # Formatting
    # =========================================================

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves - {dataset_name}", fontsize=14, fontweight="bold")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_manuscript_table(summary_df):
    table = summary_df.copy()
    table["Accuracy"] = (table["Accuracy Mean"].map(lambda x: f"{x:.4f}") + " ± " + table["Accuracy SD"].map(lambda x: f"{x:.4f}"))
    table["Precision"] = (table["Precision Mean"].map(lambda x: f"{x:.4f}") + " ± " + table["Precision SD"].map(lambda x: f"{x:.4f}"))
    table["Recall"] = (table["Recall Mean"].map(lambda x: f"{x:.4f}") + " ± " + table["Recall SD"].map(lambda x: f"{x:.4f}"))
    table["F1"] = (table["F1 Mean"].map(lambda x: f"{x:.4f}") + " ± " + table["F1 SD"].map(lambda x: f"{x:.4f}"))
    table["ROC-AUC"] = (table["ROC-AUC Mean"].map(lambda x: f"{x:.4f}") + " ± " + table["ROC-AUC SD"].map(lambda x: f"{x:.4f}"))
    table["Execution Time (s)"] = (table["Execution Time Mean"].map(lambda x: f"{x:.4f}") + " ± " + table["Execution Time SD"].map(lambda x: f"{x:.4f}"))
    table = table[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Execution Time (s)"]]
    return table