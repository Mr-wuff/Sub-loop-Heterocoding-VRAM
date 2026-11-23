# Syncing the seven audited result archives to Hugging Face

GitHub already contains a complete read-only copy of the seven SHA-256-verified experiment ZIPs at [artifacts/results](../artifacts/results/).

Companion dataset repo: https://huggingface.co/datasets/wuff-mann/Sub-loop-Heterocoding-VRAM

The Hugging Face account connection available in this ChatGPT session is read-only, so the HF repository has **not been updated by this deployment**.

To authorize the automatic upload without sharing a token in chat:

1. In Hugging Face, create a fine-grained token with **write** permission for the existing dataset repo.
2. In this GitHub repo, open **Settings → Secrets and variables → Actions → New repository secret**. Set its name to `HF_TOKEN` and its value to that token.
3. Run **Actions → Sync audited research archives to Hugging Face → Run workflow**.
4. Verify that `README.md`, `manifest.json`, and the seven `results/<round>/*.zip` files and `.sha256` sidecars appear in the dataset repository. Check the resulting commit and download one ZIP to verify SHA-256.

The workflow is manual by design: it does not silently publish to HF on every GitHub push. Existing dataset metadata currently reports Apache-2.0; the separate GitHub *software* license is not yet selected.

Local alternative after write-capable `hf auth login`:

```bash
python -m pip install huggingface_hub
python scripts/upload_hf_artifacts.py artifacts
python scripts/upload_hf_artifacts.py artifacts --upload
```

No secrets should be committed or pasted into Issues.
