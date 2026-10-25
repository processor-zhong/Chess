"""Inference on a new legal 20-ply prefix; never computes test metrics."""
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
from src.plan_a20_data import opening_features


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--white-rating',type=int,required=True)
    parser.add_argument('--black-rating',type=int,required=True)
    parser.add_argument('--base-minutes',type=int,required=True)
    parser.add_argument('--increment-seconds',type=int,default=0)
    parser.add_argument('--rated',type=int,choices=[0,1],default=1)
    parser.add_argument('--moves',required=True,help='Exactly 20 space-separated SAN moves (10 per player)')
    args=parser.parse_args()
    if min(args.white_rating,args.black_rating)<=0 or min(args.base_minutes,args.increment_seconds)<0:
        parser.error('Ratings must be positive and time controls nonnegative')
    tokens=args.moves.split()
    if len(tokens)!=20:parser.error('Provide exactly 20 plies, not the full game')
    feature,_=opening_features(tokens)
    difference=args.white_rating-args.black_rating
    pre={'rating_diff':difference,'avg_rating':(args.white_rating+args.black_rating)/2,
         'abs_rating_diff':abs(difference),'base_time_min':args.base_minutes,
         'increment_seconds':args.increment_seconds,'rated':args.rated}
    root=Path(__file__).resolve().parent
    model=joblib.load(root/'modelling_rf/run_20261025_20ply_01/models/rf_combined_final.joblib')
    X=pd.DataFrame([{**pre,**feature}])[model.feature_names_in_]
    proba=model.predict_proba(X)[0]
    prediction=str(model.classes_[proba.argmax()])
    print(json.dumps({'prediction':prediction,'probabilities':dict(zip(model.classes_,map(float,proba))),
                      'prediction_ply':20},indent=2))

if __name__=='__main__':main()
