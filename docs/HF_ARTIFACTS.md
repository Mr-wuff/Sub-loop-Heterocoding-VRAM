# Companion Hugging Face research artifacts

The benchmark archive is a **dataset-type** repository (no trained weights):

https://huggingface.co/datasets/wuff-mann/Sub-loop-Heterocoding-VRAM

Download `results/<round>/<file>.zip` together with its `.zip.sha256` sidecar. Run the CPU CLI in this repository to verify both SHA-256 and ZIP CRC; the manifest lists the original result checksums. Every experiment includes its own frozen qualification protocol. Cross-round comparisons must respect model, GPU, dtype, batch, sequence length, baseline, and peak metric.

The companion dataset is independently licensed as shown on its Hugging Face repository; the GitHub **software license is pending selection**.
