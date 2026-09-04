[![Arxiv](https://img.shields.io/badge/arXiv-2608.27394-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.27394)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-RATIO-yellow?style=flat-square)](https://huggingface.co/datasets/maayans/RATIO)
[![Website](https://img.shields.io/badge/Project-Website-blue?style=flat-square)](https://maayansharon10.github.io/RATIO/)

# RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature

[Paper](https://arxiv.org/abs/2608.27394) | [Website](https://maayansharon10.github.io/RATIO/) | [Dataset](https://huggingface.co/datasets/maayans/RATIO) | [Models](#-huggingface-repositories)

---

## What is RATIO?

RATIO (**R**etrieval **A**cross **T**yped **I**deation **O**perations) is a large-scale benchmark for **scientific inspiration retrieval**, where relevance is defined by the *ideation move* a retrieved statement enables, rather than by topical similarity. It defines three ideation moves:

- **ADDRESS** — retrieve potential approaches for a problem stated in the query.
- **BROADEN** — retrieve a more general formulation of the query.
- **SPECIFY** — retrieve a concrete instantiation of the query.

RATIO is constructed from millions of full-text CS papers via a general recipe that extends discourse-marker distant supervision — previously used only for classification — to corpus-scale retrieval, combined with extensive LLM and human vetting. Each query is paired with a single gold candidate, and all three relations share a candidate corpus of distractor sentences. The benchmark uses a strict temporal split: all test papers postdate the release of every evaluated model.

Each relation also provides a **silver test set** — a higher-precision subset of the test split, validated by human-calibrated LLM judges — for evaluation where label quality matters most.

Full details, dataset statistics, and experimental results are in the [paper](https://arxiv.org/abs/2608.27394).

<!-- TODO: add teaser figure (Figure 1 from the paper) -->
<!-- ![RATIO overview](figures/ratio_overview.png) -->

## 🤗 HuggingFace Repositories

### Dataset

| Resource | Description | Status |
| --- | --- | --- |
| [maayans/RATIO](https://huggingface.co/datasets/maayans/RATIO) | The full benchmark: `address` / `broaden` / `specify` query–gold pairs + shared `candidates` corpus. Each relation config has `train` / `validation` / `test` splits plus a `silver` split (the human-calibrated higher-precision test set) | ✅ Public |

### Models

Each base model is fine-tuned separately per relation (ADDRESS / BROADEN / SPECIFY).

| Model | When to use | Checkpoints | Status |
| --- | --- | --- | --- |
| ModernBERT-embed-large | **Recommended model** | [ADDRESS](https://huggingface.co/maayans/modernbert-embed-large__ratio-address) · [BROADEN](https://huggingface.co/maayans/modernbert-embed-large__ratio-broaden) · [SPECIFY](https://huggingface.co/maayans/modernbert-embed-large__ratio-specify) | ✅ Public |
| all-mpnet-base-v2 | Lightweight model for short texts | [ADDRESS](https://huggingface.co/maayans/all-mpnet-base-v2__ratio-address) · [BROADEN](https://huggingface.co/maayans/all-mpnet-base-v2__ratio-broaden) · [SPECIFY](https://huggingface.co/maayans/all-mpnet-base-v2__ratio-specify) | ✅ Public |
| stella_en_1.5B_v5 | Recommended for longer, instruction-based texts | [ADDRESS](https://huggingface.co/maayans/stella_en_1.5B_v5__ratio-address) · [BROADEN](https://huggingface.co/maayans/stella_en_1.5B_v5__ratio-broaden) · [SPECIFY](https://huggingface.co/maayans/stella_en_1.5B_v5__ratio-specify) | ✅ Public |
> **Input prefixes.** Each checkpoint expects the input format it was trained with:
> `all-mpnet-base-v2` — `query: ` / `document: ` · `modernbert-embed-large` — `search_query: ` / `search_document: ` · `stella_en_1.5B_v5` — built-in `s2p_query` prompt on queries (`prompt_name="s2p_query"`), no prefix on documents.
> See each model card for a ready-to-use snippet.

---

## 💻 Using Our Code

The `src/` directory is organized into three stages, each with its own README and bash run scripts so you can reproduce the full pipeline locally:

- **`src/data/`** — the dataset construction pipeline: marker lexicon construction, corpus mining from full-text papers, filtering, and the temporal split. Run these to build RATIO from scratch.
- **`src/train/`** — relation-specific contrastive fine-tuning of the retrievers. Run these to reproduce the fine-tuned checkpoints for each ideation move.
- **`src/eval/`** — retrieval evaluation and the LLM-judge top-k candidate validation used to build the silver test sets.

See the README in each folder for setup, arguments, and the exact commands to run each step.

You are welcome to use RATIO to study scientific ideation, train and evaluate retrievers, or for any other purpose. Please cite our paper as described [below](#citation).

> 🚧 **This repository is still under construction.** We are actively cleaning and documenting the pipeline. If anything is unclear or missing, you are welcome to [open an issue](https://github.com/maayansharon10/RATIO/issues) or contact the authors for additional details.

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@misc{sharon2026ratiobenchmarkretrievaltyped,
      title={RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature}, 
      author={Maayan Sharon and Tom Hope},
      year={2026},
      eprint={2608.27394},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.27394}, 
}
```

## Paper's Authors

- [Maayan Sharon](https://github.com/maayansharon10/maayansharon10)
- [Tom Hope](https://tomhoper.github.io/)