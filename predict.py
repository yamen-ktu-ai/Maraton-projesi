import argparse,csv,pathlib,json
from common import load_model,predict,read_rows
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--model',required=True);p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--output',required=True);a=p.parse_args();d=pathlib.Path(a.data);m=load_model(a.model);k=len(m['classes']);rows=read_rows(d/f'{a.split}.csv');out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    fields=['id','prediction']+[f'prob_{i}' for i in range(k)]+['latency_ms','bytes']
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            path=d/r['path'];prob,pred,ms=predict(m,path);row={'id':r['id'],'prediction':pred,'latency_ms':f'{ms:.6f}','bytes':path.stat().st_size};row.update({f'prob_{i}':f'{prob[i]:.8f}' for i in range(k)});w.writerow(row)
    print(f'{len(rows)} predictions written to {out}')
if __name__=='__main__':main()
