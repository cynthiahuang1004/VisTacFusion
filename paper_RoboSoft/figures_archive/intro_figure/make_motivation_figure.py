from __future__ import annotations
from pathlib import Path
import json, subprocess, math, shutil
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
F=ROOT/'figures/intro_figure'
ASSET=F/'motivation_assets'
W,H=520,615
C=dict(ink='242831',muted='626B76',line='BDC3CC',a='9B6254',afill='FCF8F6',aline='D8BEB5',b='3F6C91',bfill='F4F8FC',bline='C1D4E2',purple='7852BF',pfill='F6F2FC',pline='C6B1E6',orange='DF5C12',ofill='FFF1E7',oline='ECB48D',white='FFFFFF')
elems=[]
def rect(name,x,y,w,h,fill='white',stroke='line',lw=.8,r=3):
 elems.append(dict(type='rect',name=name,x=x,y=y,w=w,h=h,fill=C.get(fill,fill),stroke=C.get(stroke,stroke) if stroke else None,lw=lw,r=r))
def text(name,x,y,w,h,s,size=14,color='ink',bold=False,align='left',math=False,italic=False):
 elems.append(dict(type='text',name=name,x=x,y=y,w=w,h=h,text=s,size=size,color=C.get(color,color),bold=bold,align=align,math=math,italic=italic))
def pic(name,x,y,w,h,filename):
 im=Image.open(ASSET/filename)
 scale=min(w/im.width,h/im.height); dw,dh=im.width*scale,im.height*scale
 elems.append(dict(type='image',name=name,x=x+(w-dw)/2,y=y+(h-dh)/2,w=dw,h=dh,file=filename))
def wire(name,pts,color='muted',lw=.9,dash=False,arrow=True):
 elems.append(dict(type='wire',name=name,pts=pts,color=C.get(color,color),lw=lw,dash=dash,arrow=arrow))
# Backgrounds are intentionally flat; no drop shadows.
rect('a registration route',0,0,W,126,'afill','aline',.75,4)
rect('b appearance-rendering route',0,136,W,126,'bfill','bline',.75,4)
rect('c AlignTac route',0,276,W,339,'pfill','pline',.9,4)
# (a) Per-contact registration.
text('a title',14,9,280,22,'(a) Register real contacts',17,'a',True)
text('a bottleneck',298,10,207,21,'One registered pose per label',13.6,'a',False,'right')
pic('real acquisition photograph',24,37,66,66,'real_contact_crop.png')
wire('registration produces a labeled pair',[(101,70),(163,70)],'a')
text('registered state on acquisition arrow',111,46,40,18,r'$q^\ast$',14,'a',align='center',math=True)
pic('a same-contact real image',187,36,68,68,'real.png')
pic('a registered geometric depth',395,36,68,68,'gt.png')
text('pair plus',296,55,30,26,'+',21,'muted',align='center')
text('a photo caption',5,108,111,16,'Registered press',13,align='center')
text('a image caption',174,108,96,16,'Real image',13,align='center')
text('a geometry caption',359,108,141,16,'Registered depth',13,align='center')
# (b) Geometry is available; appearance must be fitted.
text('b title',14,145,245,22,'(b) Simulate contacts',17,'b',True)
text('b bottleneck',259,147,246,20,'Renderer fitting + appearance gap',13.4,'b',False,'right')
pic('simulated sensor-contact scene',24,173,66,66,'sim_contact.png')
pic('b same-contact simulated depth',145,175,64,64,'sim_depth.png')
rect('b alternative rendered appearances',276,169,216,91,'white','bline',.55,2)
pic('b PBR tactile rendering',284,175,64,64,'pbr.png')
pic('b learned tactile rendering',410,175,64,64,'gan.png')
wire('simulation to geometric target',[(100,207),(135,207)],'b')
wire('geometry to rendered appearance',[(219,207),(274,207)],'b')
text('render operation',219,183,55,17,'render',12.5,'b',align='center')
text('b scene caption',7,242,101,16,'Simulated contact',12.8,align='center')
text('b depth caption',136,242,82,16,'Depth target',13,align='center')
text('b pbr caption',284,242,64,16,'PBR',13,align='center')
text('b learned caption',400,242,84,16,'Learned',13,align='center')
# (c) Positive alternative and label-scaling result.
text('c title',14,287,492,23,'(c) AlignTac: real appearance + aligned geometry',16.6,'purple',True)
text('real input label',80,315,195,18,'Unlabeled real image',14,align='center')
text('geometry input label',281,315,225,18,'Known simulated geometry',14,align='center')
pic('c same-contact real image',148,339,58,58,'real.png')
pic('c known object mesh rendering',360,336,63,63,'known_geometry_stl_render.png')
# Evidence thumbnails from the supplied draft.
text('initial-depth label',10,403,126,17,'Initial geometry',13.8,align='center')
text('aligned-depth label',382,403,134,17,'Aligned geometry',13.8,'orange',align='center')
pic('mesh at teacher-estimated pose',33,425,78,78,'mesh_teacher.png')
pic('mesh at aligned contact state',409,425,78,78,'mesh_aligned.png')
rect('AlignTac operation',207,446,106,36,'white','purple',1.1,2)
text('AlignTac method name',211,452,98,25,'AlignTac',19,'purple',True,'center')
wire('initial geometry is corrected',[(115,464),(203,464)],'purple',1.1)
wire('aligned geometry becomes the pseudo-label',[(317,464),(405,464)],'orange',1.1)
wire('real appearance input',[(177,401),(177,428),(241,428),(241,443)],'purple',.85)
wire('known geometric input',[(391.5,402),(391.5,428),(279,428),(279,443)],'muted',.85)
# A small registered set supplies training support.
pic('registered real tactile training example',231,494,28,28,'registered_pair_rgb.png')
pic('registered geometric training example',263,494,28,28,'registered_pair_geometry.png')
wire('registered examples train the alignment',[(261,491),(261,485)],'purple',.95,True)
text('few registered touches label',138,526,245,18,'Few registered touches',14,'purple',align='center')
# Distinct original real RGB/aligned-depth pairs, not duplicated evidence.
wire('aligned pseudo-label route to more real contacts',[(448,507),(448,550)],'orange',.85)
wire('pseudo-label collection branch',[(82,550),(448,550)],'orange',.85,False,False)
for i,(x,cx) in enumerate([(48,81),(224,257),(400,433)],1):
 wire(f'additional labeled pair branch {i}',[(cx,550),(cx,560)],'orange',.85)
 pic(f'additional contact {i} real image',x,563,31,31,f'contact_0{i}_rgb.png')
 pic(f'additional contact {i} aligned depth',x+35,563,31,31,f'contact_0{i}_aligned_depth.png')
text('many pseudo labeled contacts label',12,596,496,18,'Many pseudo-labeled contacts for supervision',14.6,'orange',True,'center')

# LaTeX figure: same Times text and CM math as the manuscript.
def esc(s):
 return s.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
tex=['% Source: make_motivation_figure.py / motivation_layout.json. Original imagery only.','\\begingroup','\\begin{tikzpicture}[x=1bp,y=-1bp]','\\useasboundingbox (0,0) rectangle (520,615);']
for e in elems:
 if e['type']=='rect':
  opts=[f"fill={{rgb,255:red,{int(e['fill'][0:2],16)};green,{int(e['fill'][2:4],16)};blue,{int(e['fill'][4:6],16)}}}",f"rounded corners={e['r']}bp"]
  if e['stroke']:
   v=e['stroke']; opts+=[f"draw={{rgb,255:red,{int(v[0:2],16)};green,{int(v[2:4],16)};blue,{int(v[4:6],16)}}}",f"line width={e['lw']}bp"]
  else: opts+=['draw=none']
  tex.append('\\path['+','.join(opts)+f"] ({e['x']},{e['y']}) rectangle ({e['x']+e['w']},{e['y']+e['h']});")
 elif e['type']=='image':
  file='figures/intro_figure/motivation_assets/'+e['file']
  tex.append("\\node[anchor=north west,inner sep=0pt,outer sep=0pt] at (%.3f,%.3f) {\\includegraphics[width=%.3fbp,height=%.3fbp]{%s}};" % (e['x'],e['y'],e['w'],e['h'],file))
 elif e['type']=='text':
  v=e['color']; color=f"{{rgb,255:red,{int(v[0:2],16)};green,{int(v[2:4],16)};blue,{int(v[4:6],16)}}}"
  s=e['text'] if e['math'] else esc(e['text'])
  style='\\normalfont'+('\\bfseries' if e['bold'] else '')+('\\itshape' if e['italic'] else '')
  anchor={'left':'west','center':'center','right':'east'}[e['align']]
  xx=e['x']+({'left':0,'center':e['w']/2,'right':e['w']}[e['align']]); yy=e['y']+e['h']/2
  tex.append(f"\\node[anchor={anchor},inner sep=0pt,outer sep=0pt,text={color},font={{{style}\\fontsize{{{e['size']}bp}}{{{e['size']*1.12}bp}}\\selectfont}}] at ({xx:.3f},{yy:.3f}) {{{s}}};")
 elif e['type']=='wire':
  v=e['color']; opts=[f"draw={{rgb,255:red,{int(v[0:2],16)};green,{int(v[2:4],16)};blue,{int(v[4:6],16)}}}",f"line width={e['lw']}bp"]
  if e['dash']: opts.append('dash pattern=on 2.5bp off 2bp')
  if e['arrow']: opts.append('-{Stealth[length=4bp,width=3.2bp]}')
  tex.append('\\draw['+','.join(opts)+'] '+' -- '.join(f'({a:.3f},{b:.3f})' for a,b in e['pts'])+';')
tex+=['\\end{tikzpicture}%','\\endgroup']
(F/'motivation_figure.tex').write_text('\n'.join(tex)+'\n')
(F/'motivation_layout.json').write_text(json.dumps(dict(width=W,height=H,colors=C,elements=elems),indent=2))
(F/'motivation_standalone.tex').write_text(r'''\documentclass[border=1bp]{standalone}
\usepackage{fix-cm}
\usepackage{graphicx,amsmath,amssymb,xcolor,tikz}
\usetikzlibrary{arrows.meta}
\renewcommand{\rmdefault}{ptm}
\input{notation}
\begin{document}
\input{figures/intro_figure/motivation_figure}
\end{document}
''')
print('Wrote',len(elems),'elements')
