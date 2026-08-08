<!-- TODO: replace the arXiv badge + link once the preprint is live -->
[![Arxiv](https://img.shields.io/badge/arXiv-coming--soon-red?style=flat-square&logo=arxiv&logoColor=white)](#)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-RATIO-yellow?style=flat-square)](https://huggingface.co/datasets/maayans/RATIO)

# RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature

[Paper](#citation) | [Dataset](https://huggingface.co/datasets/maayans/RATIO) | [Models](#-huggingface-repositories)

---

> 🚧 **Code coming soon.** We are cleaning and documenting the full pipeline for release — see the [roadmap](#roadmap) below. The dataset is already on HuggingFace (currently private; it will be made public upon release).

## What is RATIO?

RATIO (**R**etrieval **A**cross **T**yped **I**deation **O**perations) is a large-scale benchmark for **scientific inspiration retrieval**, where relevance is defined by the *ideation move* a retrieved statement enables, rather than by topical similarity:

- **ADDRESS** — retrieve an approach or insight that responds to a problem stated in the query.
- **BROADEN** — retrieve a formulation of the query at a broader scope or greater generality.
- **SPECIFY** — retrieve a concrete instantiation or narrower formulation of the query.

RATIO is mined from millions of full-text CS papers via discourse-marker distant supervision (e.g., *"To address this issue,"*, *"More broadly,"*, *"As a concrete example,"*), combined with extensive LLM and human vetting. It contains **over 3.5M query–gold pairs** and a **shared candidate corpus of 13.8M sentences**, with a strict temporal split: all test papers postdate the release of every evaluated model.

<!-- TODO: add teaser figure (Figure 1 from the paper) -->
<!-- ![RATIO overview](figures/ratio_overview.png) -->

## Benchmark at a Glance

| Relation | Train | Validation | Test | Silver test |
| --- | ---: | ---: | ---: | ---: |
| SPECIFY | 2,605,515 | 79,583 | 94,079 | 7,327 |
| ADDRESS | 719,136 | 31,963 | 42,248 | 5,818 |
| BROADEN | 13,687 | 495 | 1,410 | 584 |
| Shared candidate corpus | 13,787,834 | 361,172 | 404,371 | — |

Each query is paired with a single gold candidate; the candidate corpus of each split is shared by all three relations and includes distractor sentences. The silver test set is a higher-precision subset validated by human-calibrated LLM judges.

## Roadmap

<!-- code coming soon -->
- [ ] **Dataset construction scripts** — marker lexicon construction, corpus mining, filtering, temporal split
- [ ] **Training scripts** — relation-specific contrastive fine-tuning of retrievers
- [ ] **Evaluation scripts** — retrieval evaluation + LLM-judge top-k candidate validation

## 🤗 HuggingFace Repositories

| Resource | Description | Status |
| --- | --- | --- |
| [maayans/RATIO](https://huggingface.co/datasets/maayans/RATIO) | The full benchmark: `address` / `broaden` / `specify` query–gold pairs + shared `candidates` corpus, in the temporal train/validation/test split | 🔒 Private — public upon release |
| ModernBERT-embed-large (fine-tuned) | Relation-specific fine-tuned checkpoints (ADDRESS / BROADEN / SPECIFY) | TODO — link coming soon |
| all-mpnet-base-v2 (fine-tuned) | Relation-specific fine-tuned checkpoints (ADDRESS / BROADEN / SPECIFY) | TODO — link coming soon |
| stella_en_1.5B_v5 (fine-tuned) | Relation-specific fine-tuned checkpoints (ADDRESS / BROADEN / SPECIFY) | TODO — link coming soon |

You are welcome to use RATIO to study scientific ideation, train and evaluate retrievers, or for any other purpose. Please cite our paper as described [below](#citation).

## Citation

If you use this code or data in your research, please cite our paper:

<!-- TODO: update eprint + url once the arXiv preprint is live -->
```bibtex
@misc{sharon2026ratio,
      title={RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature},
      author={Maayan Sharon and Tom Hope},
      year={2026},
      eprint={TODO},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={TODO},
}
```

## Authors

- [Maayan Sharon](https://github.com/maayansharon10/maayansharon10)
- [Tom Hope](https://tomhoper.github.io/)
