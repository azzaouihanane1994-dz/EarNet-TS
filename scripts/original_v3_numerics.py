"""Disclosed boundary-gradient safeguard for original ArcFace implementations."""
import ast
from pathlib import Path
import torch

def repair_angular_boundary(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ArcMarginProduct')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    class SafeSqrt(ast.NodeTransformer):
        count = 0
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch' and node.func.attr == 'sqrt':
                node.args[0] = ast.Call(func=ast.Attribute(value=node.args[0], attr='clamp_min', ctx=ast.Load()),
                    args=[ast.Constant(value=1e-7)], keywords=[])
                self.count += 1
            return node
    transformer = SafeSqrt()
    patched = transformer.visit(ast.Module(body=[fn], type_ignores=[]))
    if transformer.count != 1:
        raise RuntimeError('Unexpected original angular-head formula; refusing broad patch.')
    scope = dict(module.__dict__)
    exec(compile(ast.fix_missing_locations(patched), str(module.__file__) + ':safe-angular-boundary', 'exec'), scope)
    forward = scope['forward']
    def angular_float32(self, embeddings, *args, **kwargs):
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            return forward(self, embeddings.float(), *args, **kwargs)
    module.ArcMarginProduct.forward = angular_float32
