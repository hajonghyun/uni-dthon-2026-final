"""Standalone saved-weight E1 inference; no training/cache/data-directory dependency."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from blend_model import Predictor


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('input','image_root','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--probabilities',type=Path)
    parser.add_argument('--weights',type=Path,default=Path(__file__).resolve().parent/'weights',
                        help='Optional separate retraining output; default is the preserved #25 weights.')
    args=parser.parse_args()
    frame=pd.read_csv(args.input)
    p=Predictor(args.weights).predict(frame,args.image_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'sample_id':frame.sample_id,'label':p.argmax(-1)+1}).to_csv(args.output,index=False)
    if args.probabilities:
        args.probabilities.parent.mkdir(parents=True,exist_ok=True)
        np.save(args.probabilities,p)
    print(f'INFERENCE_COMPLETE rows={len(frame)} output={args.output}',flush=True)
