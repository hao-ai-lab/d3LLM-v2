"""Check SFT configuration/output/export agreement without loading weights or GPUs."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]


class SFTLauncherTests(unittest.TestCase):
    def test_presets_export_from_the_trainers_actual_output_directory(self):
        for preset, model in [('webshop-llada-8b', 'd3LLM_LLaDA'),
                              ('webshop-illada-8b', 'iLLaDA-8B-Instruct')]:
            with self.subTest(preset=preset), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                launcher = root / 'torchrun'
                launcher.write_text(f'#!{sys.executable}\n' + '''
import ast
from ast import literal_eval
import os
from pathlib import Path
import sys
from typing import Any, Dict, List
import yaml

index = next(i for i, arg in enumerate(sys.argv) if arg.endswith('_train.py'))
sys.argv = sys.argv[index:]
source = Path(sys.argv[0]).resolve()
__file__ = str(source)
tree = ast.parse(source.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('load_config', 'override_config')]
main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
prefix = []
for node in main.body:
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == 'training_args' for target in node.targets
    ):
        break
    prefix.append(node)
# Run the real path/configuration preparation, stopping before Trainer/model creation.
exec(compile(ast.Module(body=functions + prefix, type_ignores=[]), str(source), 'exec'))
checkpoint = Path(output_dir) / 'checkpoint-1'
checkpoint.mkdir()
(checkpoint / 'config.json').write_text('{}')
(checkpoint / 'model.safetensors').write_bytes(b'fixture')
''')
                launcher.chmod(0o755)
                env = dict(os.environ, PYTHON=sys.executable, TORCHRUN=str(launcher),
                           MODEL_PATH=str(ROOT / 'verl/models' / model),
                           TRAIN_FILE=str(ROOT / 'dLLM_sft/data/d3llm/webshop_sft_tags.parquet'),
                           OUTPUT_DIR=str(root / 'output'), EXPORT_DIR=str(root / 'export'),
                           TRAINING_TIMESTAMP='test', N_GPUS='4', USE_DEEPSPEED='0',
                           REPORT_TO='wandb', RANK='0')
                for name in ('SLURM_JOB_ID', 'REPO_ROOT', 'D3LLM_TRAIN_CONFIG'):
                    env.pop(name, None)
                result = subprocess.run(['bash', str(ROOT / 'dLLM_sft/d3LLM/run_sft.sh'), preset],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                output = root / 'output_local_test'
                config = yaml.safe_load((output / 'd3llm_train_used.yaml').read_text())
                self.assertEqual(config['training']['output_dir'], str(output))
                self.assertEqual(config['training']['report_to'], 'wandb')
                self.assertFalse(config['lora']['enabled'])
                self.assertEqual((root / 'export/model.safetensors').read_bytes(), b'fixture')


if __name__ == '__main__':
    unittest.main()
