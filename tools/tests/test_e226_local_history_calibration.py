import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).parents[1]/'scripts/run_e226_local_history_calibration.py'
s=importlib.util.spec_from_file_location('e226',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

class TestE226(unittest.TestCase):
    def test_partition_has_five_groups(self):
        a=m.groups_for([f'p{i}' for i in range(23)],'study')
        self.assertEqual(set(a.values()),set(range(5)))
        self.assertLessEqual(max(list(a.values()).count(i) for i in range(5))-min(list(a.values()).count(i) for i in range(5)),1)
    def test_partition_order_independent(self):
        p=[f'p{i}' for i in range(23)]
        self.assertEqual(m.groups_for(p,'s'),m.groups_for(p[::-1],'s'))
    def test_no_shared_groups(self):
        g=m.groups_for([f'p{i}' for i in range(23)],'s')
        for i in range(5):
            a={p for p,f in g.items() if f==i}; b={p for p,f in g.items() if f!=i}
            self.assertFalse(a&b)
if __name__=='__main__': unittest.main()
