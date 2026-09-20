# Locator weights

`model.safetensors` (474 MiB, RoBERTa-base span locator) exceeds GitHub's
100 MB file limit and is distributed as a release asset of this repository.
Download it into this directory and verify:

```bash
sha256sum -c model.safetensors.sha256
```

`../../../../preflight.py` also checks this hash before the service starts.
