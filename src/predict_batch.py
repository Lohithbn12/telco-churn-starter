import pandas as pd, joblib, sys, argparse

def main(input_csv, model_path, output_csv):
    pipe = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    proba = pipe.predict_proba(df)[:,1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out['churn_proba'] = proba
    out['churn_pred'] = pred
    out.to_csv(output_csv, index=False)
    print(f'Wrote {output_csv}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='CSV with same columns as training features')
    ap.add_argument('--model', default='models/model.joblib')
    ap.add_argument('--out', default='predictions.csv')
    args = ap.parse_args()
    main(args.inp, args.model, args.out)
