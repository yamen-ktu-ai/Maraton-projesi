"""Strict ID based evaluation. Test truth is organizer-only."""
import argparse,json,pathlib
import numpy as np
from common import read_rows
def evaluate(truth_path,pred_path,card_path):
    truth=read_rows(truth_path);pred=read_rows(pred_path);card=json.loads(pathlib.Path(card_path).read_text(encoding='utf-8'));k=len(card['classes']);multi=card['task']=='multilabel'
    ti=[r['id'] for r in truth];pi=[r['id'] for r in pred]
    if len(set(ti))!=len(ti) or len(set(pi))!=len(pi):raise ValueError('Duplicate ID')
    if set(ti)!=set(pi):raise ValueError(f'ID mismatch: missing={len(set(ti)-set(pi))}, extra={len(set(pi)-set(ti))}')
    mapped={r['id']:r for r in pred};yp=[];probs=[]
    for ident in ti:
        r=mapped[ident];p=np.array([float(r[f'prob_{c}']) for c in range(k)])
        if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid probability')
        if multi:
            raw=r['prediction'].split('|')
            if len(raw)!=k or any(v not in ['0','1'] for v in raw):raise ValueError('Invalid multilabel prediction')
            y=[int(v) for v in raw]
            # Rounding to 8 decimals permits equality ambiguity at 0.5 only.
            if any(abs(p[c]-.5)>1e-7 and y[c]!=int(p[c]>=.5) for c in range(k)):raise ValueError('Prediction differs from fixed threshold')
        else:
            y=int(r['prediction'])
            if not 0<=y<k or abs(float(p.sum())-1)>1e-5:raise ValueError('Invalid class probability')
            if p.max()-p[y]>1e-7:raise ValueError('Prediction differs from argmax')
        yp.append(y);probs.append(p)
    ytrue=np.array([[int(v) for v in r['label'].split('|')] for r in truth]) if multi else np.array([int(r['label']) for r in truth]);yp=np.array(yp)
    fs=[];per=[]
    for c in range(k):
        t=ytrue[:,c]==1 if multi else ytrue==c;q=yp[:,c]==1 if multi else yp==c;tp=int((t&q).sum());fp=int((~t&q).sum());fn=int((t&~q).sum());f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0;fs.append(f);per.append({'class':card['classes'][c],'f1':f,'tp':tp,'fp':fp,'fn':fn,'support':int(t.sum())})
    return {'track':card['track'],'count':len(ti),'macro_f1':float(np.mean(fs)),'accuracy_or_exact_match':float(np.mean(np.all(yp==ytrue,axis=1) if multi else yp==ytrue)),'per_class':per}
def main():
    p=argparse.ArgumentParser();p.add_argument('--truth',required=True);p.add_argument('--predictions',required=True);p.add_argument('--card',required=True);p.add_argument('--output');a=p.parse_args();r=evaluate(a.truth,a.predictions,a.card);text=json.dumps(r,ensure_ascii=False,indent=2);print(text)
    if a.output:pathlib.Path(a.output).write_text(text,encoding='utf-8')
if __name__=='__main__':main()
