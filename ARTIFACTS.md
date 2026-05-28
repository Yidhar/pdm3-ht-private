# Artifact Routing

Code/docs are synced to the GitHub private repository:

- https://github.com/Yidhar/pdm3-ht-private

Model artifacts/checkpoints/weights/eval outputs should be centralized in the private Hugging Face model repository:

- https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts

Do **not** commit raw ImageNet, cropped caches, PAE latent shards, training checkpoints, generated model weights, or large logs into the GitHub code repository. Keep GitHub for source/config/docs/handoff only; put model products in the HF artifacts repository when they are intentionally published.
