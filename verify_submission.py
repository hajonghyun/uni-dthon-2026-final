"""Team self-check for the five inference checks described in the code guide.

This is not the organizer's check_submission.py. All temporary data and
predictions stay in a local temporary directory and are never added to Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def check(data_dir, final, report):
    package = Path(__file__).resolve().parent
    data_dir = Path(data_dir).resolve()
    input_csv = data_dir/'test/test.csv'
    image_root = data_dir/'test'
    frame = pd.read_csv(input_csv)
    expected = pd.read_csv(final)
    if not frame.sample_id.is_unique or not expected.sample_id.is_unique:
        raise ValueError('sample_id must be unique')
    if list(expected.columns) != ['sample_id','label']:
        raise ValueError('Final CSV must contain sample_id,label')
    if not np.array_equal(frame.sample_id,expected.sample_id):
        raise ValueError('Final CSV IDs or order differ from the input')
    checks=[]
    with tempfile.TemporaryDirectory(prefix='e1-submission-check-') as temporary:
        work=Path(temporary)
        def run(name, rows, original=False):
            path=input_csv if original else work/f'{name}_input.csv'
            if not original:rows.to_csv(path,index=False)
            output=work/f'{name}_output.csv'
            command=[sys.executable,str(package/'test.py'),'--input',str(path),
                     '--image_root',str(image_root),'--output',str(output)]
            completed=subprocess.run(command,cwd=work,text=True,capture_output=True)
            if completed.returncode:
                raise RuntimeError(f'{name} inference failed: '+completed.stderr[-3000:])
            predicted=pd.read_csv(output)
            if list(predicted.columns)!=['sample_id','label'] or len(predicted)!=len(rows):
                raise ValueError(f'{name}: invalid output shape')
            if not predicted.sample_id.is_unique or not np.array_equal(predicted.sample_id,rows.sample_id):
                raise ValueError(f'{name}: output order or IDs changed')
            if not np.issubdtype(predicted.label.dtype,np.integer) or not predicted.label.between(1,6).all():
                raise ValueError(f'{name}: labels must be integers1..6')
            aligned=expected.set_index('sample_id').loc[predicted.sample_id,'label'].to_numpy()
            differences=int(np.count_nonzero(predicted.label.to_numpy()!=aligned))
            if differences:raise ValueError(f'{name}: {differences} labels differ')
            checks.append({'name':name,'rows':len(rows),'label_differences':differences})
            print(f'CHECK_PASS {name} rows={len(rows)} label_differences=0',flush=True)
            return output.read_bytes()
        full=run('full_reproduction',frame,original=True)
        if full != Path(final).read_bytes():
            raise ValueError('Reproduced final CSV bytes differ')
        if run('repeat',frame,original=True)!=full:
            raise ValueError('Repeated final CSV bytes differ')
        run('shuffle',frame.sample(frac=1,random_state=2026))
        for i,(_,rows) in enumerate(frame.groupby('experiment_id',sort=False)):
            run(f'experiment_{i+1}',rows)
        prefix=pd.concat([rows.sort_values('relative_time_s',kind='stable').iloc[:max(1,int(len(rows)*.6))]
                          for _,rows in frame.groupby('experiment_id',sort=False)],axis=0)
        run('prefix60',prefix)
        # Labels may appear in development input; they must not influence inference.
        changed=prefix.copy()
        changed['label']=6
        run('label_independence',changed)
    result={'passed':True,'organizer_checker':False,'checks':checks,
            'test_rows':len(frame),'test_experiments':int(frame.experiment_id.nunique()),
            'reproduced_final_csv_sha256':hashlib.sha256(full).hexdigest(),
            'source':'Team self-check, five guide categories plus label independence',
            'retraining':False}
    if report:
        report=Path(report);report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps(result,indent=2)+'\n')
    print('ALL_SELF_CHECKS_PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=Path,required=True)
    parser.add_argument('--final',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    check(args.data_dir,args.final,args.report)
