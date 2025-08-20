
import argparse, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from preprocess import load_and_clean, split_X_y

def main(args):
    df = load_and_clean(args.data)
    X, y = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(report)
    print("Confusion Matrix:\n", cm)
    print("ROC AUC:", round(auc, 4))

    if args.model:
        joblib.dump(clf, args.model)
        print(f"Saved model to {args.model}")

    if args.report:
        with open(args.report, 'w') as f:
            f.write(report + "\n\n")
            f.write("Confusion Matrix:\n" + str(cm) + "\n")
            f.write("ROC AUC: " + str(round(auc,4)) + "\n")
        print(f"Wrote report to {args.report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Telco churn CSV")
    parser.add_argument("--model", default="./models/churn_rf.pkl", help="Path to save model")
    parser.add_argument("--report", default="", help="Optional path to save text report")
    args = parser.parse_args()
    main(args)
