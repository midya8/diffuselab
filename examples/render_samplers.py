"""Same-model DDPM/DDIM audit: fixed shared axes, full outlier counts."""
import json
import math
import sys
from pathlib import Path


def render(r):
    p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="535" viewBox="0 0 960 535" role="img" aria-label="Same learned model sampled with DDPM and DDIM">',
       '<rect width="960" height="535" fill="white"/><g font-family="Arial,sans-serif" fill="#251f21">',
       '<text x="24" y="34" font-size="24">DiffuseLab | fewer calls, different errors</text>',
       '<text x="24" y="60" font-size="13">Same trained weights and initial seed. DDIM eta = 0. All axes: -3 to 3. 256 samples each.</text>']
    arms=[('DDPM / 64 calls',r['points']['trained'],r['metrics']['trained'],64)]
    for steps in [8,16,64]:
        a=r['samplers'][f'ddim_{steps}']
        arms.append((f'DDIM / {steps} calls',a['points'],a['metrics'],steps))
    for j,(title,points,m,calls) in enumerate(arms):
        ox,oy,side=24+234*j,105,210
        p += [f'<text x="{ox}" y="90" font-size="15" font-weight="600">{title}</text>',
              f'<rect x="{ox}" y="{oy}" width="210" height="210" rx="8" fill="white" stroke="#eae9ea"/>']
        for v in [-2,0,2]:
            u=(v+3)/6*side
            p.append(f'<path d="M{ox+u},{oy}v{side} M{ox},{oy+u}h{side}" stroke="#eae9ea"/>')
        outside=0
        for x,y in points:
            if max(abs(x),abs(y))>3:
                outside+=1;continue
            color='#73a89a' if j==0 else '#585254'
            p.append(f'<circle cx="{ox+(x+3)/6*side:.3f}" cy="{oy+(3-y)/6*side:.3f}" r="2.2" fill="{color}" opacity=".65"/>')
        lines=[f'Outside plot: {outside}/256',f'Near center: {100*m["within_0_4_of_center"]:.1f}%',
               f'Covered modes: {m["modes_with_at_least_5_nearby_samples"]}/8',
               f'Symmetric NN squared: {m["symmetric_nn_squared_distance"]:.4f}',
               f'Calls saved: {100*(1-calls/64):g}%']
        for i,line in enumerate(lines):p.append(f'<text x="{ox}" y="{338+i*21}" font-size="12">{line}</text>')
    p+=['<text x="24" y="465" font-size="13">Calls are per sample, not walltime. Near-center fraction misses large errors and frequency imbalance.</text>',
        '<text x="24" y="489" font-size="13">DDIM 8 saves 87.5% of denoiser calls but has worse symmetric reference distance here.</text>',
        '<text x="24" y="513" font-size="13">One development fixture and fixed settings. No retraining, setting search or general sampler ranking.</text>',
        '</g></svg>']
    return '\n'.join(p)+'\n'


if __name__=='__main__':
    svg=render(json.loads(Path(sys.argv[1]).read_text()))
    dest=Path('assets/sampler-audit.svg')
    if '--check' in sys.argv:assert dest.read_text()==svg,'stale sampler-audit.svg'
    else:dest.write_text(svg)
