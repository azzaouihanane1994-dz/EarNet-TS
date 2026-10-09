"""Bound evaluation memory without changing training batches or model weights."""
from functools import wraps


class Microbatches:
    def __init__(self, loader, size):
        self.loader, self.size = loader, size

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        for images, labels in self.loader:
            for start in range(0, len(images), self.size):
                yield images[start:start+self.size], labels[start:start+self.size]


def apply(module, size=8):
    for name in ('evaluate', 'extract_embeddings', 'collect_logits', 'export_visuals'):
        original = getattr(module, name, None)
        if original is None:
            continue
        def wrap(function):
            @wraps(function)
            def bounded(model, dl, *args, **kwargs):
                return function(model, Microbatches(dl, size), *args, **kwargs)
            return bounded
        setattr(module, name, wrap(original))
