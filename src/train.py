import pandas as pd, joblib, json, argparse, os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from pipeline import build_pipeline

def main(data_path, model_dir):
    df = pd.read_csv(data_path)
    # Basic cleaning: drop blank TotalCharges rows and cast
    df = df[df['TotalCharges'].ne(" ")]
    df['TotalCharges'] = df['TotalCharges'].astype(float)

    target_col = 'Churn'
    y = (df[target_col] == 'Yes').astype(int)
    X = df.drop(columns=[target_col, 'customerID'])

    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','bool']).columns.tolist()

    pipe = build_pipeline(num_cols, cat_cols)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring='roc_auc', n_jobs=-1)
    pipe.fit(X_tr, y_tr)

    y_hat = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    report = classification_report(y_te, y_hat, output_dict=True)
    auc = roc_auc_score(y_te, y_proba)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(model_dir, 'model.joblib'))
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump({'cv_auc_mean': float(scores.mean()), 'cv_auc_std': float(scores.std()), 'test_auc': float(auc), 'report': report}, f, indent=2)

    print(f"Model saved to {os.path.join(model_dir, 'model.joblib')}\nTest AUC: {auc:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    parser.add_argument('--out', default='models')
    args = parser.parse_args()
    main(args.data, args.out)
