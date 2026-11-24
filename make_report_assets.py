"""Generate report tables/plots from recorded checks and actual inference outputs."""
from pathlib import Path
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
results = ROOT / 'results'
report = ROOT / 'report'
(report / 'figures').mkdir(parents=True, exist_ok=True)
pattern = re.compile(r'^(\w+)\s+(PASS|FAIL) shape=(\([^\n]+?\)) max_abs=([\deE+.\-]+) mean_abs=([\deE+.\-]+) \(atol=([\deE+.\-]+), rtol=([\deE+.\-]+)\)', re.M)
rows = []
for m in pattern.finditer((results / 'check_all.txt').read_text()):
    stage, status, shape, maximum, mean, atol, rtol = m.groups()
    rows.append(dict(stage=stage,status=status,shape=shape,max_abs=float(maximum),mean_abs=float(mean),atol=float(atol),rtol=float(rtol)))
assert len(rows) == 7 and all(r['status']=='PASS' for r in rows)
(results / 'stage_metrics.json').write_text(json.dumps(rows, indent=2)+'\n')
def sci(value):
    if value == 0: return '$0$'
    coefficient, exponent = f'{value:.3e}'.split('e')
    return '$'+coefficient+r'\times10^{'+str(int(exponent))+'}$'
tex=[]
for r in rows:
    name=r['stage'].replace('_',r'\_')
    tex.append(r'\code{'+name+'} & '+r'\code{'+r['shape'].replace(' ','')+'} & '+sci(r['max_abs'])+' & '+sci(r['mean_abs'])+r' & PASS \\')
(report / 'stage_rows.tex').write_text('\\newcommand{\\StageRows}{%\n'+'\n'.join(tex)+'\n}\n',encoding='utf-8')
with np.load(results/'output_tokens.npz', allow_pickle=False) as out:
    actual, ids = out['output_tokens'], out['ids']
with np.load(ROOT/'assets/reference_outputs.npz', allow_pickle=False) as ref:
    assert np.array_equal(ids, ref['ids'])
    error = np.abs(actual-ref['output_tokens'])
per_image = [{'id':str(i),'max_abs':float(e.max()),'mean_abs':float(e.mean())} for i,e in zip(ids,error)]
(results/'per_image_metrics.json').write_text(json.dumps(per_image,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8.0,2.6))
x=np.arange(len(ids));width=.34
ax.bar(x-width/2,error.max(1)*1e7,width,color='#B42128',label='Maximum absolute error')
ax.bar(x+width/2,error.mean(1)*1e7,width,color='#4A748F',label='Mean absolute error')
ax.set_xticks(x,[str(i).replace('image_','') for i in ids])
ax.set_xlabel('Supplied image ID');ax.set_ylabel('Absolute error (x 1e-7)')
ax.set_ylim(0,7);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
ax.legend(frameon=False,ncol=2,loc='upper center',fontsize=9)
fig.tight_layout(pad=.9)
for extension in ['pdf','png']:
    fig.savefig(report/'figures'/('output_errors.'+extension),dpi=220,bbox_inches='tight')
plt.close(fig)
print('Generated report/stage_rows.tex and report/figures/output_errors.pdf/png')
