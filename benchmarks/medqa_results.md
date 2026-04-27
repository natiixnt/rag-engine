# MedQA / BioASQ Evaluation

Biomedical retrieval evaluation across two complementary benchmarks: MedQA (USMLE-style multiple-choice questions requiring evidence retrieval) and BioASQ (biomedical literature search). The corpus is a domain-tuned biomedical literature index built on PubMed abstracts plus full-text from PMC Open Access.

## Headline Numbers

| Metric                        | rag-engine | Notes |
|-------------------------------|-----------|-------|
| Retrieval Accuracy@10         | **0.87**  | Biomedical literature retrieval (BioASQ 11b) |
| MedQA Evidence R@10           | 0.84      | USMLE-style evidence retrieval |
| BioASQ MAP                    | 0.731     | BioASQ Task 11b document retrieval |
| BioASQ Snippet F1             | 0.642     | Sentence-level snippet relevance |
| Mean Avg Precision (top-100) | 0.703      | Combined PubMed retrieval |

The 0.87 retrieval accuracy on biomedical literature is on the BioASQ 11b test set. We sit between general-domain rag-engine (~0.82 on biomedical) and the heavy specialized systems like PubMedBERT-MedQA fine-tuned variants (~0.89-0.91), at a fraction of their inference cost.

## Eval Setup

### Corpus
- **34.7M biomedical documents**
- Composition: 32.1M PubMed abstracts (1990-2025), 2.4M PMC full-text articles, 200K clinical guidelines (NICE, USPSTF, AHA, etc.)
- Avg abstract length: 247 tokens, full text: 4382 tokens (chunked at 512 with 64 overlap, parent at 2048)
- Embedding model: `BAAI/bge-base-en-v1.5` for the headline number, also tested with PubMedBERT-base (results below)

### Query sets
- **BioASQ 11b**: 1,498 expert-annotated biomedical questions with ground-truth document and snippet relevance labels
- **MedQA-Evidence**: 1,273 USMLE-style questions where we measure whether the retrieved passages contain the supporting evidence (annotated by 2 MD reviewers)
- **MIMIC-IV-CMQA**: 740 clinical multi-question queries from de-identified clinical notes (we use this for the "real clinical workflow" eval, separate from the academic benchmarks)

### Pipeline config
- Default rag-engine v0.5.0 pipeline
- HyDE *enabled* for biomedical (unlike legal). LLM generates hypothetical abstracts which align well with PubMed style. Boosts recall@10 by 2.1 pts on short queries.
- Cross-encoder fine-tuned on 120K biomedical query-passage pairs (BioASQ training + internal annotated). +6.7 MRR vs general-domain reranker.
- Acronym expansion preprocessor: "MI" -> "myocardial infarction (MI)" to bridge the medical jargon gap. Adds 0.3ms, gives +1.1 R@10.

## Detailed Results

### BioASQ 11b Document Retrieval

| Metric                   | rag-engine | PubMedBERT + reranker | BM25 baseline |
|--------------------------|-----------|------------------------|----------------|
| Recall@10                | 0.87      | 0.89                   | 0.71           |
| MAP                      | 0.731     | 0.748                  | 0.512          |
| MRR                      | 0.812     | 0.834                  | 0.624          |
| GMAP                     | 0.658     | 0.671                  | 0.421          |

We sit ~1.7 MAP points behind the best PubMedBERT-tuned system but run 14x faster at p95.

### MedQA Evidence Retrieval

USMLE-style questions where the answer requires citing supporting evidence from the literature.

| Question Type            | R@10  | MRR   |
|--------------------------|-------|-------|
| Diagnosis                | 0.86  | 0.81  |
| Treatment                | 0.88  | 0.84  |
| Pathophysiology          | 0.81  | 0.76  |
| Pharmacology             | 0.85  | 0.79  |
| Anatomy                  | 0.79  | 0.74  |
| Lab interpretation       | 0.83  | 0.78  |
| **Average (1273 q's)**   | 0.84  | 0.79  |

Pathophysiology and anatomy lag because the supporting evidence is often in textbooks (less covered in PubMed) rather than primary literature.

### BioASQ Snippet Retrieval

Sentence-level snippets within the retrieved documents. Harder than document retrieval because it requires fine-grained evidence localization.

| Metric                   | rag-engine | Best baseline (BioBERT-NER) |
|--------------------------|-----------|-------------------------------|
| Snippet Precision        | 0.687     | 0.703                          |
| Snippet Recall           | 0.604     | 0.621                          |
| Snippet F1               | 0.642     | 0.658                          |

We trail by 1.6 F1 on snippets. Late interaction helps here but the cross-encoder isn't trained at sentence-level granularity. Possible v0.6.0 work: train a snippet-extraction head on the reranker.

## Embedding Model Ablation

Tested with three embedding models on BioASQ 11b:

| Embedding model           | R@10  | MRR   | Index build time |
|---------------------------|-------|-------|------------------|
| `BAAI/bge-base-en-v1.5`   | 0.87  | 0.81  | 47 min            |
| `PubMedBERT-base`         | 0.89  | 0.84  | 52 min            |
| `MedCPT-Article-Encoder`  | 0.91  | 0.86  | 1h 18min          |

`MedCPT` (NCBI's biomedical CPT model) gives the best retrieval quality but takes 65% longer to encode the corpus. For most production deployments the bge-base + domain-finetuned reranker combination is the better quality-per-dollar trade. We expose the embedding model as a config knob; teams with biomedical-only workloads typically swap to MedCPT.

## Comparison to Specialized Systems

| System                              | R@10  | MRR   | p95 latency | Notes |
|-------------------------------------|-------|-------|-------------|-------|
| BM25 only                           | 0.71  | 0.62  | 3.4 ms      | Strong baseline on PubMed |
| BioBERT + sparse fusion             | 0.79  | 0.71  | 28.7 ms     | Common research config |
| PubMedBERT + cross-encoder          | 0.89  | 0.84  | 247 ms      | Heavy but high quality |
| MedCPT + reranker                   | 0.91  | 0.86  | 198 ms      | NCBI's pipeline |
| **rag-engine (default)**            | 0.87  | 0.81  | 19.4 ms     | Our default config |
| **rag-engine (MedCPT swap)**        | 0.91  | 0.86  | 22.7 ms     | Our config + MedCPT embeddings |

The MedCPT swap closes the quality gap to NCBI's pipeline at ~9x lower latency. This is the deployment story for biomedical customers who care about both quality and throughput.

## Clinical Workflow Notes

Biomedical retrieval has some quirks worth calling out:

- **Negation matters a lot**: "patient has myocardial infarction" vs "patient does not have myocardial infarction" should retrieve different evidence. Standard embeddings handle this poorly. We added a negation-aware preprocessor that tags negated entities and the cross-encoder learned to weight them. +0.7 MRR on negation-heavy queries.
- **Acronym ambiguity**: "MS" can be multiple sclerosis, mitral stenosis, mass spectrometry, or morphine sulfate depending on context. Disambiguation is handled by the embedding model when the query is long enough; on short queries we lean on metadata filters (specialty, document type) where available.
- **Temporal relevance**: Medical knowledge evolves. The 1995 paper on a treatment may be superseded by 2023 guidelines. We surface publication date as metadata and bias rerank scores toward recency for clinical guideline queries (configurable, off by default for primary literature).
- **PHI safety**: For clinical-note retrieval (MIMIC-IV use case), HyDE is *off* by default because we don't want patient identifiers leaking to an external LLM. On-prem LLM (Llama 3.1 70B served locally) is the supported config there.

## Reproducibility

```bash
# Public benchmarks (BioASQ 11b, MedQA)
python benchmarks/run_biomedical.py \
    --datasets bioasq_11b medqa_evidence \
    --corpus /data/biomedical_34m \
    --embeddings bge-base-en-v1.5 \
    --output benchmarks/medqa_results.json
```

PubMed + PMC corpus is freely downloadable from NCBI. Our exact corpus (with internal preprocessing for chunking) is reproducible via `scripts/build_biomedical_corpus.py` (not committed, available on request).

## Limitations on Biomedical

- No multi-language support. Non-English biomedical literature (Chinese, Japanese, German registries) is on the v0.7.0 list.
- Image-heavy modalities (radiology, pathology) need vision models for full retrieval. We're text-only.
- We don't model patient-specific context. Personalized retrieval (matching evidence to specific patient demographics, comorbidities) is a clinical decision support problem outside our scope.
- Drug-drug interaction queries are weak. The cross-encoder doesn't have strong DDI training data. We rely on structured DDI databases (downstream, not in retrieval).
