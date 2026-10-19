#!/bin/zsh
# Recreates ref/ (untracked, read-only upstream reference): agens-engine at b0e9a95 and the release's code and
# config files. Benchmarks, the parity tools and spec/make_suite.py read from it.
set -e
cd "${0:A:h}"
HF=${HF:-${0:A:h}/../.venv/bin/hf}
mkdir -p ref
[[ -d ref/agens-engine ]] || git clone https://github.com/BlockWayz/agens-engine ref/agens-engine
git -C ref/agens-engine checkout -q b0e9a95
$HF download Blockway/Agens-Volundr-32B-v1-INT4 --local-dir ref/hf-int4 --include \
  modeling_volundr.py bcsa.py bcsa_flex.py volundr_extras.py config.json generation_config.json \
  model.safetensors.index.json tokenizer_config.json agens_b_tokens.json MODIFICATIONS.md README.md SHA256SUMS
$HF download Blockway/Agens-Volundr-32B-v1-Engine-16GB --local-dir ref/hf-16gb --include \
  config.json quantization_audit.json README.md
