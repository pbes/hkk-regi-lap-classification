import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

CSV_INPUT = "card_flags.csv"
CSV_OUTPUT_PREDICTIONS = "card_flags_predicted.csv"

ALL_FLAGS = [
    "token", "visszavétel", "semmizés", "lopás", "dobatás",
    "gyógyulás", "sebzés", "erőforrás", "keresés", "húzás",
    "reakció", "leszedés",
]

TRAIN_TEST_THRESHOLD = 9405

HUNGARIAN_STOPWORDS = [
    "a", "az", "és", "is", "de", "hogy", "nem", "ez", "azt", "meg",
    "ha", "egy", "be", "ki", "le", "el", "fel", "van", "vagy", "volt",
    "lesz", "csak", "már", "még", "mint", "sem", "se", "majd",
    "igen", "igen", "mind", "minden", "mindig", "más", "itt", "ott",
    "úgy", "így", "akkor", "mikor", "ahol", "ami", "aki", "amely",
    "amit", "itt", "ott", "ide", "oda", "te", "ő", "mi", "ti", "ők",
    "én", "neki", "őt", "ezt", "azt", "stb", "ill", "pl", "vagyis",
    "tehát", "illetve", "hiszen", "pedig", "mert", "után", "előtt",
    "által", "között", "mellett", "alatt", "fölött", "rajta", "benne",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_INPUT, encoding="utf-8")
    df["text"] = df["text"].fillna("")
    df["flags"] = df["flags"].fillna("")
    df["is_reaction"] = df["is_reaction"].astype(bool)
    return df


def encode_flags(df: pd.DataFrame) -> np.ndarray:
    """Convert pipe-separated flags string to binary matrix."""
    y = np.zeros((len(df), len(ALL_FLAGS)), dtype=int)
    for i, flags_str in enumerate(df["flags"]):
        if flags_str:
            for flag in flags_str.split("|"):
                flag = flag.strip()
                if flag in ALL_FLAGS:
                    y[i, ALL_FLAGS.index(flag)] = 1
    return y


def build_features(df: pd.DataFrame, vectorizer: TfidfVectorizer, fit: bool = False):
    """Build feature matrix from text (TF-IDF) + is_reaction."""
    if fit:
        text_features = vectorizer.fit_transform(df["text"])
    else:
        text_features = vectorizer.transform(df["text"])

    is_reaction = csr_matrix(df["is_reaction"].astype(int).values.reshape(-1, 1))
    return hstack([text_features, is_reaction])


def main():
    df = load_data()

    # Split by ID threshold
    train_test_df = df[df["id"] >= TRAIN_TEST_THRESHOLD].reset_index(drop=True)
    predict_df = df[df["id"] < TRAIN_TEST_THRESHOLD].reset_index(drop=True)

    print(f"Train/test set: {len(train_test_df)} cards")
    print(f"Prediction set: {len(predict_df)} cards")

    # Encode labels
    y = encode_flags(train_test_df)

    # Print distribution
    flags_per_card = y.sum(axis=1)
    print("\n--- Flag count distribution (train/test set) ---")
    for n in range(int(flags_per_card.max()) + 1):
        count = (flags_per_card == n).sum()
        print(f"  {n} flags: {count} cards ({100 * count / len(y):.1f}%)")

    print("\n--- Per-flag distribution (train/test set) ---")
    for i, flag_name in enumerate(ALL_FLAGS):
        count = y[:, i].sum()
        print(f"  {flag_name}: {count} cards ({100 * count / len(y):.1f}%)")

    # TF-IDF on text
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words=HUNGARIAN_STOPWORDS)
    X = build_features(train_test_df, vectorizer, fit=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining: {X_train.shape[0]} samples, Testing: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")

    # --- Compare algorithms ---
    candidates = {
        "LogisticRegression": OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=1.0, random_state=42), n_jobs=-1
        ),
        "LinearSVC": OneVsRestClassifier(
            LinearSVC(max_iter=2000, C=1.0, random_state=42), n_jobs=-1
        ),
        "RandomForest": OneVsRestClassifier(
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        ),
        "XGBoost": OneVsRestClassifier(
            XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                eval_metric="logloss", random_state=42,
            ),
            n_jobs=-1,
        ),
        "LightGBM": OneVsRestClassifier(
            LGBMClassifier(
                n_estimators=200, learning_rate=0.1, random_state=42, verbose=-1,
            ),
            n_jobs=-1,
        ),
    }

    print("\n--- Algorithm comparison ---")
    results: dict[str, float] = {}
    for name, clf in candidates.items():
        print(f"  Training {name}...", end=" ", flush=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        results[name] = f1
        print(f"macro-F1: {f1:.4f}")

    best_name = max(results, key=results.__getitem__)
    best_clf = candidates[best_name]
    print(f"\nBest model: {best_name} (macro-F1={results[best_name]:.4f})")

    y_best_pred = best_clf.predict(X_test)
    print(f"\n--- Classification Report: {best_name} ---")
    print(classification_report(y_test, y_best_pred, target_names=ALL_FLAGS, zero_division=0))

    # Predict on old cards (ID < 9405) using best model
    X_predict = build_features(predict_df, vectorizer, fit=False)
    predictions = best_clf.predict(X_predict)

    # Decode predictions back to flag strings
    predicted_flags = []
    for row in predictions:
        flags = [ALL_FLAGS[i] for i, v in enumerate(row) if v == 1]
        predicted_flags.append("|".join(flags) if flags else "")

    predict_df = predict_df.copy()
    predict_df["predicted_flags"] = predicted_flags

    predict_df.to_csv(CSV_OUTPUT_PREDICTIONS, index=False, encoding="utf-8")
    print(f"\nPredictions written to {CSV_OUTPUT_PREDICTIONS}")

    # Summary
    flag_counts = predict_df["predicted_flags"].apply(
        lambda x: len(x.split("|")) if x else 0
    )
    print(f"\nPrediction summary ({len(predict_df)} cards):")
    print(f"  Cards with 0 flags: {(flag_counts == 0).sum()}")
    print(f"  Cards with 1 flag: {(flag_counts == 1).sum()}")
    print(f"  Cards with >1 flags: {(flag_counts > 1).sum()}")
    print(f"  Average flags per card: {flag_counts.mean():.2f}")


if __name__ == "__main__":
    main()
