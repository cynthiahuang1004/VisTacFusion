"""Export Figure 1's editable PowerPoint from motivation_layout.json.

Requirements: python-pptx, PyMuPDF; pdfLaTeX for the mathematical label.
Most labels are native editable text. Math is embedded as SVG with a PNG
fallback, with its editable LaTeX source in motivation_math/.
"""
from __future__ import annotations
from pathlib import Path
import json, subprocess
from lxml import etree
from pptx import Presentation
from pptx.util import Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
import fitz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=json.loads((HERE/'motivation_layout.json').read_text())
assets=HERE/'motivation_assets'
mathdir=HERE/'motivation_math'
mathdir.mkdir(exist_ok=True)
prs=Presentation()
prs.slide_width=Pt(spec['width'])
prs.slide_height=Pt(spec['height'])
slide=prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid();slide.background.fill.fore_color.rgb=RGBColor(255,255,255)

def no_effects(sh):
    spPr=sh._element.spPr
    for old in list(spPr):
        if etree.QName(old).localname in ['effectLst','effectDag']:
            spPr.remove(old)
    spPr.append(OxmlElement('a:effectLst'))
    for ref in sh._element.xpath('.//a:effectRef'):
        ref.set('idx','0')

def rgb(s): return RGBColor.from_string(s)

def math_image(e,i):
    # Use the same unmodified CM math family as main.tex / notation.tex.
    content=r'''\documentclass[border=0.8bp]{standalone}
\usepackage{fix-cm,amsmath,amssymb,xcolor}
\renewcommand{\rmdefault}{ptm}
\input{notation}
\begin{document}
{\fontsize{SIZEbp}{LEADbp}\selectfont\color[HTML]{COLOR}FORMULA}
\end{document}
'''.replace('SIZE',str(e['size'])).replace('LEAD',str(e['size']*1.12)).replace('COLOR',e['color']).replace('FORMULA',e['text'])
    tex=mathdir/f'label_{i:02d}.tex';tex.write_text(content)
    r=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(mathdir),str(tex)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if r.returncode: raise RuntimeError(r.stdout[-4000:])
    pdf=fitz.open(tex.with_suffix('.pdf')); page=pdf[0]
    svg=mathdir/f'label_{i:02d}.svg';png=mathdir/f'label_{i:02d}.png'
    svg.write_text(page.get_svg_image(text_as_path=True))
    page.get_pixmap(matrix=fitz.Matrix(5,5),alpha=True).save(png)
    dw,dh=page.rect.width,page.rect.height
    xx=e['x']+(e['w']-dw)/2; yy=e['y']+(e['h']-dh)/2
    p=slide.shapes.add_picture(str(png),Pt(xx),Pt(yy),Pt(dw),Pt(dh))
    part=Part(PackURI(f'/ppt/media/formula{i}.svg'),'image/svg+xml',slide.part.package,svg.read_bytes())
    rid=slide.part.relate_to(part,RT.IMAGE)
    blip=p._element.blipFill.blip
    extlst=OxmlElement('a:extLst'); ext=OxmlElement('a:ext');ext.set('uri','{96DAC541-7B7A-43D3-8B79-37D633B846F1}')
    svgblip=etree.Element('{http://schemas.microsoft.com/office/drawing/2016/SVG/main}svgBlip',nsmap={'asvg':'http://schemas.microsoft.com/office/drawing/2016/SVG/main'})
    svgblip.set('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed',rid)
    ext.append(svgblip);extlst.append(ext);blip.append(extlst)
    p.name=e['name'];no_effects(p)

for i,e in enumerate(spec['elements']):
    typ=e['type']
    if typ=='rect':
        sh=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if e['r'] else MSO_SHAPE.RECTANGLE,Pt(e['x']),Pt(e['y']),Pt(e['w']),Pt(e['h']))
        if e['r']:
            sh.adjustments[0]=min(0.5,e['r']/min(e['w'],e['h']))
        sh.fill.solid();sh.fill.fore_color.rgb=rgb(e['fill'])
        if e['stroke']:
            sh.line.color.rgb=rgb(e['stroke']);sh.line.width=Pt(e['lw'])
        else: sh.line.fill.background()
        sh.name=e['name'];no_effects(sh)
    elif typ=='image':
        sh=slide.shapes.add_picture(str(assets/e['file']),Pt(e['x']),Pt(e['y']),Pt(e['w']),Pt(e['h']))
        sh.name=e['name'];no_effects(sh)
    elif typ=='text':
        if e['math']:
            math_image(e,i);continue
        sh=slide.shapes.add_textbox(Pt(e['x']),Pt(e['y']),Pt(e['w']),Pt(e['h']))
        sh.name=e['name'];tf=sh.text_frame;tf.clear()
        tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
        tf.auto_size=MSO_AUTO_SIZE.NONE;tf.word_wrap=False;tf.vertical_anchor=MSO_ANCHOR.MIDDLE
        p=tf.paragraphs[0];p.alignment={'left':PP_ALIGN.LEFT,'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT}[e['align']]
        p.space_before=p.space_after=Pt(0);p.line_spacing=1.0
        r=p.add_run();r.text=e['text'];r.font.name='Times New Roman';r.font.size=Pt(e['size']);r.font.color.rgb=rgb(e['color']);r.font.bold=e['bold'];r.font.italic=e['italic']
        no_effects(sh)
    elif typ=='wire':
        for j,(p1,p2) in enumerate(zip(e['pts'],e['pts'][1:])):
            sh=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Pt(p1[0]),Pt(p1[1]),Pt(p2[0]),Pt(p2[1]))
            sh.name=e['name']+f' / segment {j+1}'
            sh.line.color.rgb=rgb(e['color']);sh.line.width=Pt(e['lw'])
            if e['dash']:sh.line.dash_style=MSO_LINE_DASH_STYLE.DASH
            ln=sh._element.spPr.get_or_add_ln()
            if e['arrow'] and j==len(e['pts'])-2:
                a=OxmlElement('a:tailEnd');a.set('type','triangle');a.set('w','sm');a.set('len','sm');ln.append(a)
            no_effects(sh)
notes=slide.notes_slide.notes_text_frame
notes.text=('AlignTac Figure 1: labeling bottlenecks and geometric alignment. '
            'The photograph, simulation scene, same-contact RGB / reference depth, PBR and learned renderings, '
            'teacher-initialized depth, and aligned depth are supplied artifacts from _draft__RoboSoft2027(2).zip. '
            'The geometry rendering, registered-pair icons and additional RGB-depth pairs are retained from the accepted revised figure. '
            'The registration panel and rendering panel illustrate alternatives, not sequential stages. PBR and Learned are alternative renderers. '
            'The dashed arrow is training support from registered contacts. The method is schematic; no experimental values were changed. '
            'Boxes, arrows, and plain labels are native editable PowerPoint elements. The q-star label is vector artwork with editable LaTeX in motivation_math. '
            'The manuscript uses the corresponding TikZ / PDF figure with Times text and CM math. No fonts are embedded in this PPTX.')
prs.core_properties.title='AlignTac: labeling bottlenecks and aligned supervision'
prs.core_properties.author=''
target=HERE/'AlignTac_Figure1_Updated.pptx'
prs.save(target)
# Remove theme shadow fallbacks as well, for consistent PDF exports in LibreOffice.
from zipfile import ZipFile, ZIP_DEFLATED
with ZipFile(target) as z:
    contents={name:z.read(name) for name in z.namelist()}
for name in list(contents):
    if name.startswith('ppt/theme/') and name.endswith('.xml'):
        tree=etree.fromstring(contents[name])
        ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
        for effect in tree.xpath('//a:effectStyleLst/a:effectStyle',namespaces=ns):
            for child in list(effect): effect.remove(child)
            effect.append(etree.Element('{'+ns['a']+'}effectLst'))
        contents[name]=etree.tostring(tree,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(target,'w',ZIP_DEFLATED) as z:
    for name,data in contents.items():z.writestr(name,data)
print('Created:',HERE/'AlignTac_Figure1_Updated.pptx')
