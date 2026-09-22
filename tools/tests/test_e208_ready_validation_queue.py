import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
p=Path(__file__).parents[1]/'scripts/run_e208_ready_validation_queue.py'
s=importlib.util.spec_from_file_location('ready',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)


class ReadyValidation(unittest.TestCase):
    def test_only_complete_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); d=root/'latent/seed_1'; d.mkdir(parents=True)
            self.assertIsNone(m.ready_checkpoint(root,'latent',1))
            state=d/'E208_RUN_STATUS.json'
            state.write_text(json.dumps({'status':'RUNNING'}))
            self.assertIsNone(m.ready_checkpoint(root,'latent',1))
            ckpt=d/'best.ckpt'; ckpt.write_bytes(b'test')
            row={'status':'COMPLETE','architecture':'latent','seed':1,'best_checkpoint':str(ckpt),
                 'test_perturbed_expression_rows_read':0}
            state.write_text(json.dumps(row)); self.assertEqual(m.ready_checkpoint(root,'latent',1),ckpt)
            row['test_perturbed_expression_rows_read']=1; state.write_text(json.dumps(row))
            with self.assertRaises(m.queue.PostTrainingFailure): m.ready_checkpoint(root,'latent',1)


if __name__=='__main__': unittest.main()
