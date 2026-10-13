#!/usr/bin/env python3
"""Validate the LRX obligation DAG and render all current views, offline/std-lib."""
import argparse, hashlib, html, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'projects/lrx'
STATES={'backlog':'Очередь','working':'Работа','review':'Проверка','integration':'Интеграция','done':'Готово','blocked':'Блокер'}
def validate(d):
 nodes=d['nodes']; ids=[n['id'] for n in nodes]; assert len(set(ids))==len(ids),'Duplicate id'
 by={n['id']:n for n in nodes}
 for n in nodes:
  assert n['state'] in STATES
  assert n['owner'] and n['reviewer'] and n['acceptance'] and n['statement']
  assert set(n['requires'])<=set(ids),'Unknown dependency'
  assert n['assignment'] in ['proposed','accepted','completed']
  if n['state']=='done':assert n['evidence'] and n['claim']=='proved' and n['verification']!='not-reviewed','Unsupported done'
 seen=set(); active=set()
 def visit(k):
  assert k not in active,'Cycle'
  if k in seen:return
  active.add(k)
  for dep in by[k]['requires']:visit(dep)
  active.remove(k);seen.add(k)
 for k in ids:visit(k)
 assert 'ALT-C' not in by['EQ']['requires'],'Alternative incorrectly mandatory'
 if by['EQ']['state']=='done':
  assert all(by[k]['state']=='done' for k in seen if by[k]['strategy']=='main'),'Equality before prerequisites'
 return d

def render(d):
 digest=hashlib.sha256((P/'proof-map.json').read_bytes()).hexdigest(); ns=d['nodes']
 head=f"Срез свидетельств: {d['evidence_cutoff']}. Версия {d['revision']}. SHA-256 реестра: `{digest}`.\n\n{d['freshness']}\n\n"
 roadmap='# LRX: текущая карта доказательства\n\n'+head+d['goal']+'\n\n'+d['status_note']+'\n\n'
 roadmap+='Основной путь: MODEL + U-SOURCE → LIFT/BRIDGE → PLAN + EXEC + CAP → COST/SMALL → U-ALL; D + U-ALL → EQ. Это зависимости применения, не запрет параллельной работы. ALT-C — альтернативный путь; RELEASE — отдельная публикационная задача.\n\n'
 roadmap+='| Узел | Статус пакета | Ответственный / принятие | Зависимости |\n|---|---|---|---|\n'
 for n in ns:roadmap+=f"| [{n['id']}](#{n['id'].lower()}) · {n['title']} | {STATES[n['state']]} | {n['owner']} / {n['assignment']} | {', '.join(n['requires']) or '—'} |\n"
 for n in ns:
  roadmap+=f"\n## {n['id']}\n\n{n['statement']}\n\n- Утверждение: {n['claim']}; проверка: {n['verification']}; публикация: {n['publication']}.\n- Приёмка: {n['acceptance']}\n- Следующий шаг: {n['next_action']}\n- Reviewer: {n['reviewer']}. Checkpoint: {n['next_checkpoint'] or 'новый срок не подтверждён'}.\n- Issue: {n['issue'] or 'см. общую координацию #3'}.\n- Свидетельства: {', '.join(n['evidence']) or 'обязательство; результата пока нет'}.\n"
 kanban='# LRX: текущий канбан\n\n'+head+'Состояние пакета не равно доказанности всей гипотезы.\n'
 for state,label in STATES.items():
  kanban+='\n## '+label+'\n\n'
  for n in ns:
   if n['state']==state:kanban+=f"- **{n['id']} · {n['title']}** — {n['owner']} ({n['assignment']}). {n['next_action']}\n"
 payload=json.dumps(d,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
 template=(P/'dashboard-template.html').read_text()
 page=template.replace('__DATA__',payload).replace('__HASH__',digest)
 return {'ROADMAP.md':roadmap.rstrip()+'\n','KANBAN.md':kanban.rstrip()+'\n','dashboard.html':page}
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--check',action='store_true');args=a.parse_args()
 output=render(validate(json.loads((P/'proof-map.json').read_text())))
 for name,text in output.items():
  if args.check:assert (P/name).exists() and (P/name).read_text()==text, f'Stale generated view: {name}'
  else:(P/name).write_text(text)
 print('PASS: obligation DAG, evidence gates, 3 synchronized views')
