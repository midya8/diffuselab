"""Render the real demo samples. All four panels use the SAME coordinate scale."""
import json
import math
import sys
from pathlib import Path


def render(result):
    colors = {'reference':'#585254','trained':'#73a89a','gaussian_prior':'#585254','untrained':'#585254'}
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="500" viewBox="0 0 960 500" role="img" aria-label="Reference, learned DDPM and controls on identical axes">',
           '<rect width="960" height="500" fill="white"/>',
           '<g font-family="Arial,sans-serif" fill="#251f21">',
           '<text x="24" y="34" font-size="24">DiffuseLab | learn the reverse process</text>',
           '<text x="24" y="59" font-size="13">Eight synthetic modes. 256 samples per panel. Same axes: x and y from -3 to 3.</text>']
    names=[('reference','Reference fixture'),('trained','Learned DDPM'),('gaussian_prior','Gaussian prior'),('untrained','Untrained DDPM')]
    for j,(key,title) in enumerate(names):
        ox=24+j*234; oy=103; side=210
        parts += [f'<text x="{ox}" y="89" font-size="15" font-weight="600">{title}</text>',
                  f'<rect x="{ox}" y="{oy}" width="{side}" height="{side}" rx="8" fill="white" stroke="#eae9ea"/>']
        for u in [-2,0,2]:
            pos=(u+3)/6*side
            parts += [f'<path d="M{ox+pos},{oy}v{side} M{ox},{oy+pos}h{side}" stroke="#eae9ea"/>']
        for k in range(8):
            x=ox+(2*math.cos(2*math.pi*k/8)+3)/6*side
            y=oy+(3-2*math.sin(2*math.pi*k/8))/6*side
            parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="14" fill="none" stroke="#c0bebf"/>')
        outside=0
        for x,y in result['points'][key]:
            if max(abs(x),abs(y))>3:
                outside+=1;continue
            parts.append(f'<circle cx="{ox+(x+3)/6*side:.3f}" cy="{oy+(3-y)/6*side:.3f}" r="2.2" fill="{colors[key]}" opacity=".65"/>')
        parts.append(f'<text x="{ox}" y="335" font-size="12">Outside plot: {outside}/256</text>')
        if key!='reference':
            m=result['metrics'][key]
            parts += [f'<text x="{ox}" y="358" font-size="12">Near a center: {100*m["within_0_4_of_center"]:.1f}%</text>',
                      f'<text x="{ox}" y="379" font-size="12">Covered modes: {m["modes_with_at_least_5_nearby_samples"]}/8</text>',
                      f'<text x="{ox}" y="400" font-size="12">Symmetric NN squared: {m["symmetric_nn_squared_distance"]:.4f}</text>']
    parts += ['<text x="24" y="441" font-size="13">Near = within 0.4. Covered = at least 5 nearby samples. Counts do not imply balanced mode frequencies.</text>',
              '<text x="24" y="464" font-size="13">One development fixture. No seed search, image model, calibrated density or general performance claim.</text>',
              '</g></svg>']
    return '\n'.join(parts)+'\n'


if __name__=='__main__':
    result=json.loads(Path(sys.argv[1]).read_text())
    svg=render(result)
    dest=Path('assets/diffusion-audit.svg')
    if '--check' in sys.argv:
        assert dest.read_text()==svg,'stale diffusion-audit.svg'
    else:
        dest.write_text(svg)
