# GitHub Release Notes

This file records the public release constraints for ActPro.

## Public Resources

- Model adapters: [Jambo1988/ActLLaVA](https://huggingface.co/Jambo1988/ActLLaVA)
- ASTime dataset: [Jambo1988/ASTime](https://huggingface.co/datasets/Jambo1988/ASTime)
- PKUMMD dataset: [ECHO960/PKU-MMD](https://github.com/ECHO960/PKU-MMD)

## Files Kept Out Of GitHub

The GitHub repository should contain code, annotations, configuration files,
documentation, and small metadata files. Large binaries and generated artifacts
should stay outside GitHub.

Do not commit raw adapter weights:

```text
Act-LLaVA/checkpoints/ActLLaVA-ASTime/adapter_model.safetensors
Act-LLaVA/checkpoints/ActLLaVA-PKUMMD/adapter_model.safetensors
```

These files are hosted through the Hugging Face model repository above.

Do not publish pre-extracted ASTime or PKUMMD training features. Users should
generate them locally with the preprocessing scripts. The expected feature
locations are:

```text
Act-LLaVA/dataset/ASTime/features/<video_key>.pt
Act-LLaVA/dataset/PKUMMD/features/<video_key>.pt
```

Generated predictions, GPT caches, temporary sampled videos, Python caches,
local `.env` files, and experiment output folders should also stay out of the
GitHub upload unless they are intentionally documented release results.

## Model Names

Use these adapter names consistently in the repository:

```text
ActLLaVA-ASTime
ActLLaVA-PKUMMD
```

The old experiment-specific names should not appear in public instructions.

## License

The repository is released under Apache License 2.0. The base models, datasets,
and third-party dependencies are governed by their own licenses and terms.
