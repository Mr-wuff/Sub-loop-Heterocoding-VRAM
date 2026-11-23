# Reproduction pointers

1. Read the GitHub code README and `docs/REPRODUCTION.md` once the public links are live.
2. Select the same experiment ID and inspect the result ZIP's frozen protocol/environment before running its source notebook.
3. Start from the same base GPT-2 checkpoint and data revision where supplied by the experiment. Do not infer quality from matching result metadata alone.
4. Run the corresponding Colab notebook in a clean GPU session, export a new ZIP and compare its raw fields with this archive.
5. Report any CUDA, precision, data or software version difference explicitly.

All artifacts are small, original result JSON/CSV ZIPs. No model checkpoint, proprietary input text, API token, or raw third-party training dataset is included.
