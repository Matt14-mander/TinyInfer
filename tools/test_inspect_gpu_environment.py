"""Inventory resolution checks; no GPU, compiler or system changes required."""
import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch

import inspect_gpu_environment as inventory


class ToolkitResolutionTest(unittest.TestCase):
    def resolve(self, located=None, environment=None, exists=False):
        output = io.StringIO()
        with patch.dict(os.environ, environment or {}, clear=True), \
                patch('sys.argv', ['inspect_gpu_environment']), \
                patch.object(inventory.shutil, 'which', return_value=located), \
                patch.object(inventory.platform, 'system', return_value='Windows'), \
                patch.object(inventory.Path, 'is_file', return_value=exists), \
                patch.object(inventory, 'probe', side_effect=lambda command: {
                    'status': 'mock', 'command': command}), \
                contextlib.redirect_stdout(output):
            inventory.main()
        return json.loads(output.getvalue())

    def test_path_takes_precedence(self):
        result = self.resolve('C:/path/nvcc.exe', {'CUDA_PATH': 'C:/other'})
        self.assertEqual(result['cuda_toolkit_resolution'], 'PATH')
        self.assertEqual(result['cuda_toolkit']['command'][0], 'C:/path/nvcc.exe')

    def test_cuda_path_fallback(self):
        result = self.resolve(environment={'CUDA_PATH': 'C:/cuda'}, exists=True)
        self.assertEqual(result['cuda_toolkit_resolution'], 'CUDA_PATH')
        self.assertEqual(inventory.Path(result['cuda_toolkit']['command'][0]),
                         inventory.Path('C:/cuda/bin/nvcc.exe'))

    def test_missing_sdk(self):
        result = self.resolve()
        self.assertEqual(result['cuda_toolkit_resolution'], 'not_found')

    def test_invalid_cuda_path_is_not_reported_as_installed(self):
        result = self.resolve(environment={'CUDA_PATH': 'C:/missing'})
        self.assertEqual(result['cuda_toolkit_resolution'], 'not_found')


if __name__ == '__main__':
    unittest.main()
