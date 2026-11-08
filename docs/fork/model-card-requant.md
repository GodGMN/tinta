---
license: cc-by-4.0
base_model: nvidia/parakeet-tdt-0.6b-v3
tags:
  - automatic-speech-recognition
  - parakeet
  - onnx
  - sherpa-onnx
  - int8
  - tinta
library_name: onnx
---

# Parakeet TDT 0.6B v3, int8 encoder (Tinta distribution)

Draft model card for a future Hugging Face repository
(`GodGMN/parakeet-tdt-0.6b-v3-int8-tinta`). **Nothing has been created or
uploaded.** It is the README the repository should carry when it exists.

This is a re-quantized **encoder** for NVIDIA's Parakeet TDT 0.6B v3, packaged
with the sherpa-onnx int8 decoder, joiner and tokens so one revision pins the
whole transducer. It is the default transcription artifact of Tinta's
Windows/Linux port (task #23), where the shipped int8 export dropped whole
stretches of speech.

## Attribution

- **Original model:** Parakeet TDT 0.6B v3, © NVIDIA,
  <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>, licensed under
  **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>).
- **ONNX export:** `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3` at revision
  `5fcc0122fe87e8365e339e467af90c67b746218f` (the fp32 encoder and the fp32
  decoder/joiner/tokens), and its int8 sibling for the decoder, joiner and
  tokens as packaged here.
- **Modification:** the encoder was **re-quantized**. MatMul-only, per-channel
  signed int8 weights with `reduce_range=True`; Conv, LayerNormalization and the
  embeddings stay fp32; the sherpa-onnx `metadata_props` are copied from the
  fp32 encoder. The decoder, joiner and tokens are unchanged copies of the
  sherpa-onnx int8 export.
- NVIDIA does not endorse Tinta or this redistribution.

## Files

| File | Bytes | SHA-256 |
|---|---:|---|
| `encoder.int8.onnx` | 881,596,689 | `23c100540a23ac714ece37ba69c2c59bad9fc70ab1e0bd5b368f98d71a2c336f` |
| `decoder.int8.onnx` | 11,845,275 | `179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e` |
| `joiner.int8.onnx` | 6,355,277 | `3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3` |
| `tokens.txt` | 93,939 | `d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d` |

Total: 899,797,241 B (858 MiB).

## Recipe

Input: the fp32 sherpa-onnx export, pinned by SHA-256:

```
36b758364dfddce472b9a9f75bf6049bbd35c886d7927e3eda80d4ceb3348512  encoder.onnx
3af3f51af5f2d01dbbf5af47d42c7962a2c205f11004254bb4f2b979862f39a8  encoder.weights
```

```python
from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic(
    "encoder.onnx", "encoder.int8.onnx",
    op_types_to_quantize=["MatMul", "Gemm"],  # Conv stays fp32
    per_channel=True,
    reduce_range=True,                        # non-VNNI x86 (U8S8 saturation)
    weight_type=QuantType.QInt8,
    use_external_data_format=False,           # 841 MiB fits one file
)
```

The sherpa-onnx model metadata (`model_type=EncDecRNNTBPEModel`, `vocab_size`
8192, `subsampling_factor` 8, ...) is copied from the fp32 encoder. The
reproduction script is `scripts/fork/segmentation/requantize-encoder.py` in
`GodGMN/tinta`; its quantizer pins are `onnxruntime==1.31.1`, `onnx==1.23.2`
and `numpy==2.4.6`.

`reduce_range=True` is required on x86 CPUs without AVX-512 VNNI / AVX-VNNI:
the U8S8 `MatMulInteger` kernel there accumulates into a saturating int16, and
full-range per-channel int8 weights overflow it. With `reduce_range` the 7-bit
weights fit (255·64·2 = 32,640 < 32,767). VNNI and ARM dotprod accumulate in
int32 and are unaffected.

## Measured results

Parakeet TDT 0.6B v3, Tinta engine, 2 threads. Lower is better.

| Test | int8 (shipped) | this encoder | fp32 |
|---|---|---|---|
| 13.4 s AMI window 235.4-248.8 s | 0 words | 31 words (full text) | 31 words |
| AMI IHM 1,000 utterances, WER | 14.94 % | **13.16 %** | 13.79 % |
| tcpWER, 3 meetings, 5 s collar | 25.7 % | **20.2 %** | 20.3 % |
| tcpWER, 3 meetings, 1 s collar | 31.4 % | **26.2 %** | 26.4 % |
| Engine peak RSS, worst meeting | 2.26 GiB | **3.00 GiB** | 4.21 GiB |

This encoder is **equivalent to fp32 within noise** (at least as good); the
gain over the shipped int8 file is recovered deletions at the cost of extra
insertions. See `docs/fork/asr-experiments.md` in `GodGMN/tinta` for the full
method, confidence intervals and per-meeting numbers.

The portability CI job rebuilds the encoder with the pinned quantizer on x64
AVX2, x64 VNNI, Linux arm64 and Windows arm64, verifies the hash and decodes a
committed clip (at least 28 words, at most 10 % normalized WER against the fp32
text).

## Usage

sherpa-onnx transducer (`nemo_transducer`), with the four files above in one
directory:

```python
import sherpa_onnx

recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder="encoder.int8.onnx",
    decoder="decoder.int8.onnx",
    joiner="joiner.int8.onnx",
    tokens="tokens.txt",
    model_type="nemo_transducer",
    num_threads=2,
)
```

Tinta pins the same files by SHA-256 in
`crates/tinta-engine-rs/assets/models.json` and downloads them from a pinned
revision.

## License

CC BY 4.0. You may share and adapt the model, including commercially, provided
you credit NVIDIA, link the license, state the modification described above,
and do not imply endorsement. No additional restrictions are placed on this
redistribution.
