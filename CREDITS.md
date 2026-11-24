# Credits

## Model

**NVIDIA**: [Nemotron 3.5 Lightning 30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
(revision a9904d24), released under [OpenMDW-1.1](https://openmdw.ai/license/1-1/). The quantized weights are a
derivative of this model and keep its license.

## Engine

| Who | Role | License | Source |
|---|---|---|---|
| **Ash (ashhart) and the TensorFold contributors** | TensorFold, the engine this recipe's fork builds on | Apache-2.0 (code before 0.6.0 keeps its MIT notice) | [ashhart/TensorFold](https://github.com/ashhart/TensorFold) |
| **vcruz305** | TensorFold's EXL3 module, the base of the expert decoders | Apache-2.0 | TensorFold |
| **turboderp** | ExLlamaV3 and the EXL3 format | MIT | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) |

The engine's `THIRD_PARTY_NOTICES.md` holds the complete list of upstream work and licenses.

## Recipe

Kristian Aaron, with Claude (Anthropic) as engineering assistant.
