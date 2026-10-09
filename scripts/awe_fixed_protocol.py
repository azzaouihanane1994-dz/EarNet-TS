"""Enforce the author-supplied AWE lists; seeds never alter membership."""
import ast
import hashlib
import json
from pathlib import Path

def apply_fixed_awe(module, destination, root):
    base = root / 'data/awe'
    lists = {name: list(map(int, (base / (name + '.txt')).read_text().split())) for name in ['train', 'test']}
    assert len(lists['train']) == 600 and len(lists['test']) == 400
    assert len(set(lists['train'])) == 600 and len(set(lists['test'])) == 400
    assert not set(lists['train']) & set(lists['test'])
    assert set(lists['train']) | set(lists['test']) == set(range(1, 1001))
    items = {}
    for name, indices in lists.items():
        items[name] = []
        for index in indices:
            pid = f'{(index - 1) // 10 + 1:03d}'
            image = base / pid / f'{(index - 1) % 10 + 1:02d}.png'
            if not image.exists(): raise FileNotFoundError(image)
            items[name].append((image.relative_to(root), pid))
    labels = {f'{i:03d}': i - 1 for i in range(1, 101)}
    # Preserve all 600 supplied training images. The legacy val loader is only a
    # training-set monitor; final checkpoint is predetermined by epoch budget.
    module.split_dataset = lambda cfg: (items['train'], items['train'], items['test'], labels)
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8-sig'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    class FinalCheckpoint(ast.NodeTransformer):
        count = 0
        def visit_If(self, node):
            self.generic_visit(node)
            if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == 'va_acc' and any(isinstance(x, ast.Name) and x.id == 'best_val_acc' for x in node.test.comparators):
                node.test = ast.Compare(left=ast.Name(id='epoch', ctx=ast.Load()), ops=[ast.Eq()],
                    comparators=[ast.Subscript(value=ast.Name(id='cfg', ctx=ast.Load()), slice=ast.Constant('epochs'), ctx=ast.Load())])
                self.count += 1
            return node
    patcher = FinalCheckpoint()
    patched = patcher.visit(ast.Module(body=[main], type_ignores=[]))
    if patcher.count != 1: raise RuntimeError('Unknown checkpoint selection; AWE execution refused.')
    exec(compile(ast.fix_missing_locations(patched), str(module.__file__) + ':fixed-awe', 'exec'), module.__dict__)
    audit = {'protocol': 'author-supplied fixed AWE split', 'train_images':600, 'test_images':400,
             'seed_changes_membership':False, 'validation_role':'Training-set monitor only; no independent validation claim.',
             'checkpoint_selection':'Final predetermined epoch; test images never select checkpoint or settings.',
             'file_sha256':{name:hashlib.sha256((base/(name+'.txt')).read_bytes()).hexdigest() for name in lists}}
    (destination / 'awe_fixed_protocol.json').write_text(json.dumps(audit, indent=2))
