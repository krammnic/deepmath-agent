import copy, importlib.util, json, unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('renderer',P/'scripts/render_lrx_map.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class MapTests(unittest.TestCase):
 def setUp(self): self.d=json.loads((P/'projects/lrx/proof-map.json').read_text())
 def test_current(self):m.validate(self.d)
 def test_cycle(self):
  self.d['nodes'][0]['requires']=['EQ']
  with self.assertRaises(AssertionError):m.validate(self.d)
 def test_unknown_dependency(self):
  self.d['nodes'][0]['requires']=['MISSING']
  with self.assertRaises(AssertionError):m.validate(self.d)
 def test_unsupported_done(self):
  n=next(n for n in self.d['nodes'] if n['id']=='PLAN');n['state']='done'
  with self.assertRaises(AssertionError):m.validate(self.d)
 def test_premature_equality(self):
  n=next(n for n in self.d['nodes'] if n['id']=='EQ');n.update(state='done',claim='proved',verification='formally-checked',evidence=['test'])
  with self.assertRaises(AssertionError):m.validate(self.d)
 def test_alternative_not_mandatory(self):
  next(n for n in self.d['nodes'] if n['id']=='EQ')['requires'].append('ALT-C')
  with self.assertRaises(AssertionError):m.validate(self.d)
if __name__=='__main__':unittest.main()
