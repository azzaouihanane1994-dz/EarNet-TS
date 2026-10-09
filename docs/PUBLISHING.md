# Publishing

The repository is initialized locally with branch `main`. The public tree contains code, documentation, dataset manifests, AWE membership lists, saved embeddings, score arrays, matching full-model metrics, and relevant figures. It excludes images, checkpoint binaries, generated runs and interpreter caches.

After creating a repository on GitHub, configure its URL and push from this directory:

```bash
git remote add origin https://github.com/OWNER/REPOSITORY.git
git add .
git commit -m "Add full-model code and experimental evidence"
git push -u origin main
```

Run `python scripts/check_release.py` before publication. Check its file hashes and article metric comparisons. Files from separate experiments should be added only with their actual logs, scoring definitions, memberships and configurations. The supplied public archive follows the same image/checkpoint exclusions as Git.

The software license does not grant rights to distribute third-party datasets or pretrained weights. Keep dataset acquisition links available; distribute eligible large artifacts separately with matching hashes and their own terms.
