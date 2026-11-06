# Internal review, round 3: triage

Status: 2026-11-06. Line numbers refer to `paper_submission/sections/*.tex` (and `main.tex`) in the current working tree and will drift as we edit.
The original review is reproduced word for word in the [Appendix](#appendix-original-review-verbatim).

## Applied so far (2026-11-06)

Applied: M4.3, M4.4, M4.5, M5.1 (simplified, without the caveat), M5.2 (∼90× real-valued, plus a note in the Table S6 caption), M6.1, M6.2, M6.3, M6.5, M6.6, M6.8, M7.1, M7.2 (λ = 0), M7.5, M8.1, M8.4, M8.5, M8.6.
Second batch: M1.1 ("for the descriptors we fit"), M1.2, M2.1 (worded as "plays the role of the coupling", plus one caveat sentence after the Hybrid equation), M2.2 (short version), M2.3 (dropped "mechanistic" instead of the longer replacement), M3.1 (Corollary made conditional on r⋆ ≥ 1/(ρN), without the empirical numbers).
Not touched, per your decision: everything else.
Open: whether to rerun for M4.1. A no-rerun route is possible, but Methods l.43 must be corrected either way.
M3.2 decided: the main FCD/phFCD training loss is now the sorted-entry MSE (1-D W₂²), not a control row. Code: `sorted_entry_mse` in [dynamics_metrics.py](src/metrics/dynamics_metrics.py), used by `FCD`/`PhFCD` for both the loss and the logged `*_mse`. New result JSONs carry `"dynamics_loss": "sorted_entry_mse"`. Paper: Results l.35, Methods (metrics table, FCD formula plus a short explanation, phFCD), and the Supplement bound section (sorted vectors, W₂ remark, one-way remark moved to the opening paragraph, stale commented paragraph removed). Weights unchanged (all 1.0); on empirical segments the sorted loss is about 0.3–0.6× the entrywise one.
**Rerun required for every trained result** (Tables 1, S1, S2, S4, S5, Figs 2c, 3, 4); bundle it with the M4.1 rerun. The paper text already describes the new loss, but every number is still from the old loss.

## How to read the tables

| Code | Meaning | Effect on length |
|---|---|---|
| **FIX** | The statement is wrong or overstated, so we correct it | ±0 |
| **CLARIFY** | A protocol fact a referee needs is missing; we add at most 1–2 sentences | + a few lines |
| **KEEP** | Leave as is; the justification is in the table | 0 |
| **RERUN** | A correctness problem: recompute the same tables | 0 (numbers change) |
| **OPTIONAL** | A new analysis, only if we agree it is worth the space | + |

Policy: correct everything that is wrong, and add the protocol facts a referee needs to reconstruct the experiments. Add no new analyses unless a claim cannot stand without them. Where the reviewer offers "either run X or soften the claim", we soften.

---

## 0. What the code check found (read this first)

I checked these points against the repository, not only the PDF.

1. **The connectome prior C₀ and the frequencies ω include the test subjects in the main comparison.** The reviewer did not raise this, and it is the most important item in this document (see [M4.1](#m4--information-separation)). The main N = 94 runs ([logs/revision/launch_seeds.sh](logs/revision/launch_seeds.sh)) do not pass `--train-subset`. As a result, C₀ is the `FC_mean` stored in the `.mat` file: the mean Pearson FC of the **raw, full 1200-volume recordings of all 94 subjects**, validation and test included. ω is also estimated from all subjects. [4_methods.tex:43](paper_submission/sections/4_methods.tex#L43) states the opposite, and the docstring at [train_models.py:685-691](examples/train_models.py#L685-L691) already acknowledges the issue.
2. **ψ_out(0, z_i) = 0 is not enforced.** `OutputNetwork` is a biased MLP on `[m_i, z_i]`; only its last layer starts near zero ([hybrid_hopf_model.py:93-110](src/models/hybrid_hopf_model.py#L93-L110)). The claim that the model "learns only the coupling" is therefore not guaranteed.
3. **The FCD and phFCD training losses are entry-by-entry MSE** between the simulated and empirical matrices ([dynamics_metrics.py:425-436](src/metrics/dynamics_metrics.py#L425-L436)). The reviewer's M3 objection applies.
4. **Preprocessing runs on the whole recording before the T/2 split.** z-scoring, the FFT band-pass (brick-wall, non-causal, circular) and the Hilbert transform are all applied first ([data_loader.py:438-447](src/dataset/data_loader.py#L438-L447)). This affects only within-subject evaluations.
5. **The Neural SDE widths in the Methods text are wrong; Table S6 is right.** The config uses drift 256/256 and diffusion 128, which gives 168,648 + 51,528 = **220,176** parameters.
6. **Coupled Hopf:** the gradient fit learns 2 parameters (a, G; κ = 1, ω and σ fixed). The grid scans 3 (G × a × κ = 5 × 4 × 2).
7. **R(Θ) is unused.** `loss_weights["l2"] = 0.0`, so the complexity regularizer in the Methods objective is never active.
8. **GNN-Hopf widths are wrong in the Supplement.** It says ψ_θ has "2 hidden layers, width 32"; `GNNHopfConfig` uses **1 layer, width 16**. The reviewer did not flag this.
9. **Autocorrelation:** in Table S2 the Hybrid is significantly *worse* than both comparators. Our reply to comment 7 was wrong.
10. **The Hybrid's 2,215 parameters count each complex weight once.** In real degrees of freedom it has 2,428, so the ratio to the Neural SDE is about 91, not about 99.

---

## A. The eight comments from the previous round (R0–R7)

| # | Topic | Reviewer's position now | Decision | What we do / why we leave it |
|---|---|---|---|---|
| R0 | Independent prediction | Can be deferred | **KEEP** | It needs new data, and the reviewer accepts deferring it. The scope is set by the wording change in M1.1: the paper is about statistical reproduction, and interventions stay prospective. |
| R1 | Physiological interpretability | Partly addressed | **FIX** (via M2.3) | Narrow the wording only. The reviewer agrees that no new experiments are needed. |
| R2 | External validation | Can be deferred | **CLARIFY** | Add one clause to the Discussion: one healthy HCP cohort; an internal subject split is not external validation. The paper does not currently say this anywhere. |
| R3 | Central contribution | Substantially addressed | **KEEP** | See M1.3–M1.4. |
| R4 | Static vs. dynamic personalization | Resolved | **KEEP** | Nothing to do. |
| R5 | Surrogate alignment | Important and still open | **CLARIFY** (+ OPTIONAL) | See M3.2. "Not done" is not an acceptable reply; we need at least to state what the bound does not show. |
| R6 | Small neural baseline | Reconsider | **FIX** | See M5.1. Restrict the claim to the architecture and budget we tested; no new runs. |
| R7 | Autocorrelation | Must be corrected | **FIX** | Our reply was wrong: Table S2 marks both comparisons as significant. See M6.1–M6.2. |

---

## B. The eight required-change blocks (M1–M8)

### M1 — State exactly what is learned

| # | Request | Where / verified | Decision | Action or justification |
|---|---|---|---|---|
| M1.1 | Frame the work as generative fitting of selected descriptors, not as recovery of the true brain dynamics | Intro l.23 ("learns only the component…"), Discussion | **FIX** | Reword one clause in the Intro, combined with M2.1. No new paragraph. |
| M1.2 | Replace "fully interpretable" | Intro l.27 | **FIX** | Change to "has few parameters with direct biophysical readings". |
| M1.3 | Position the paper against the four closest antecedents | Intro l.14–21 | **KEEP** | The Intro already contrasts us with physics-based, data-driven and hybrid work. A paper-by-paper comparison would amount to a related-work section, which the format does not need. At most, add one clause naming the closest antecedent. |
| M1.4 | Put a number in the abstract (94 subjects, composite improvement, non-uniform gain) | `main.tex` l.143 | **KEEP** (or swap) | The abstract is at 148/150 words. "94 HCP participants" and "most balanced … across static and dynamic descriptors" already give the sample and signal that the gain is not uniform. The numbers are in Table 1 and the first Results paragraph. We decided last round not to add a result sentence. If co-authors want a number, a word-neutral swap is "lowest composite error (0.086 vs. 0.129 and 0.182)" in place of "most balanced data fidelity", using the values after the M4.1 rerun. |
| M1.5 | Parameter accounting in the abstract | — | **KEEP** | Not needed, and it depends on M5.2/M7.1 anyway. |

### M2 — Correct the scope of the coupling restriction

| # | Request | Where / verified | Decision | Action or justification |
|---|---|---|---|---|
| M2.1 | "Learns only the coupling" holds only if ψ_out(0, z_i) = 0 | **Verified: the constraint is not imposed** (finding 2) | **FIX** | Do **not** change the model. Reword Intro l.23, Results (architectures paragraph) and Discussion l.14/18 to: "retains an explicit Hopf local term and learns graph-conditioned interactions; because ψ_out also receives the local state, the learned term is not formally restricted to inter-regional effects." Add the same caveat after the Hybrid equation (Methods l.324). |
| M2.2 | The Supplement wrongly says GNN-Hopf differs by using the receiver's state | Supplement l.267 ("pointwise nonlinearity…") | **FIX** | Replace it with the real differences. GNN-Hopf has no key encoder (it aggregates raw states), and it uses a full n×n C instead of the rank-10 LRᵀ. Also correct its widths (M7.5). |
| M2.3 | The learned C is not an identified physiological influence matrix; "mechanistic interpretability" is overstated | Discussion l.18 ("partially interpretable as a data-driven effective connectivity"), l.20 ("mechanistic interpretability") | **FIX** | Replace "mechanistic interpretability" with "interpretability of the retained explicit components". Add one sentence: the effect of region j on region i also depends on φ_key and ψ_out, and LRᵀ is not unique, so C is a structured prior, not an identified connectivity. The commented-out identifiability paragraph in the Discussion has a sentence ready to reuse. This also covers R1. |

### M3 — Resolve the objection to the surrogate loss

| # | Request | Where / verified | Decision | Action or justification |
|---|---|---|---|---|
| M3.1 | Corollary: because the empirical distribution has atoms, Q_b(r) ≤ ρr fails at small r; the Corollary must require r⋆ to lie in the valid range | Supplement l.46: **the scope paragraph is already drafted but commented out** | **FIX** | Uncomment it after re-checking its numbers (Q_b(r)/r < 14 for r ≥ 0.01, median ≈ 5). State the Corollary as conditional on r⋆ ≥ 1/(ρN). Almost no cost. |
| M3.2 | Entry-by-entry MSE between independent FCD realizations rewards lower simulated variability; the bound does not justify pairing entries | **Verified** (finding 3). The objection is valid and mirrors our own argument against trajectory MSE (Methods l.60–64). A referee will notice. | **CLARIFY** (+ OPTIONAL) | **Minimum:** add 3–4 sentences to the Supplement. The bound runs one way and says nothing about whether pairing entries is well calibrated. For decorrelated realizations, E‖A−B‖² = ‖EA−EB‖² + Var A + Var B. The shared initial condition and short windows (100 TRs, about 20 FCD windows) keep the pairing partly meaningful. We therefore treat the MSE as a heuristic surrogate and evaluate only with KS on held-out data. **Optional, to close the point fully:** one extra row in the existing ablation table (S4): the Hybrid trained with sorted-entry MSE, which is the 1-D empirical W₂ and a two-line change (sort the upper-triangle entries before the MSE). 3 seeds at about 45 GPU-min each; no new section or figure. Risk: if it beats the entry-by-entry loss, we must explain why the main model keeps the entry-by-entry loss. |
| M3.3 | Permutation diagnostic without retraining | — | **KEEP** (skip) | The reviewer calls it a diagnostic, not a substitute. It adds content without closing the point. |

### M4 — Make the information separation unambiguous

| # | Request | Where / verified | Decision | Action or justification |
|---|---|---|---|---|
| M4.1 | **Source of the prior:** say which subjects and segments build C₀ and ω | **Verified: C₀ and ω use all 94 subjects (test included), full raw recordings** (finding 1). Methods l.43 says the opposite. | **RERUN** (recommended) | See the box below. |
| M4.2 | Table or diagram of the experiments (subjects, segments, prior, ω, checkpoint selection, evaluation set) | — | **OPTIONAL** (merge with M7.4) | If we add one structural element this round, it should be a single supplementary table, "Experimental protocols", that serves both M4 and M7. Rows: main comparison, sample size, personalization, ablations, grid vs. gradient. It replaces scattered text and fixes the inconsistencies in one place. Otherwise, CLARIFY inline. |
| M4.3 | The Supplement says "N = 94 fits", but population training uses 65 subjects | Supplement l.86 | **FIX** | Change to "cohort-level fits (65 training subjects)". |
| M4.4 | Is preprocessing done before or after the split? | **Verified: before** (finding 4) | **CLARIFY** | Add one sentence to the Methods. The inter-subject evaluations (Table 1, S2, ablations) are unaffected: test subjects are never seen, and preprocessing uses no cross-subject statistics. Only the within-subject evaluations are affected (personalization, and any `test_intra` numbers). No rerun is needed. Personalization leaves a 150-volume gap (volumes 450–600) between the training and test segments, and its effect (ΔFC corr ≈ +0.2 for 10/10 targets) is far larger than any boundary effect of filtering could produce. |
| M4.5 | Personalization: describe it as calibration conditioned on a population prior and say which data build that prior | **Verified:** `train_personalized.py --sc-source group` uses `dataset.fc_mean`, built from all 94 subjects' full recordings, including each target's held-out half | **CLARIFY** | Add one sentence: the prior is shared by all ten fits, so it cancels in Δ = own − others, which is why we use that contrast. Wording: "individual calibration conditioned on a population connectome prior". |
| M4.6 | 10-subject experiment: is the prior re-estimated from the 10 subjects or taken from the full cohort? | **Verified:** with `--train-subset`, C₀ and ω come from the 10 training subjects only (`_training_cohort`), but from their full recordings | **CLARIFY** + **CHECK** | State it. **Check on the cluster** whether the 65-subject bars in Fig. 3d come from the main runs (all-94 prior) or from `--train-subset 65` runs. If they come from the main runs, the 10-vs-65 comparison mixes two prior protocols; the M4.1 rerun fixes this automatically. |
| M4.7 | HCP subset: unrelated subjects? Session, phase-encoding direction, selection criteria; family-aware split | The repository cannot answer this | **CLARIFY** (needs co-author input) | Add one sentence to the Data section. If the 94 subjects come from the unrelated-subjects release, say so. If families are present, acknowledge it as a limitation rather than re-splitting everything. |

> **M4.1 — the prior includes the test subjects.**
>
> *Size of the effect* (seeds 42–44; the split was reproduced exactly with `compute_split_indices` in a scratch script):
> - Correlation between the all-94 prior and a training-only prior: r = 0.998–0.999 using full recordings, and 0.993–0.995 using the training subjects' first halves.
> - Correlation of the test subjects' mean FC with the prior: 0.976–0.982 for the all-94 prior vs. 0.966–0.975 for the training-only prior. Including the test subjects therefore adds **+0.007 to +0.010**.
> - For scale, the Table 1 FC correlations are Hybrid 0.926, Neural SDE 0.921 and Coupled Hopf 0.913: gaps of the same size.
> - Only the Hopf-family models use C₀; the Neural SDE does not. The advantage is small, but it is **asymmetric and in our favour**, on the metric family where the gaps between models are smallest.
>
> *Why rerun rather than justify:* this is a correctness problem, not added content. The tables keep their shape and the text barely changes. The code path already exists (`_training_cohort` in [train_models.py](examples/train_models.py)); it only needs to run when `--train-subset` is absent.
> - **Minimum scope:** Coupled Hopf and Hybrid Hopf × 9 seeds (Table 1).
> - **Full scope:** also GNN-Hopf and Hopf+Neural (S2), the Hopf-family ablations (S4) and the grid search (S11), which all use C₀.
>
> *Fallback if we do not rerun:* correct Methods l.43 to describe what was actually done, report the numbers above, and acknowledge the asymmetry. That is defensible, but a referee who reads it will probably ask for the rerun anyway.

### M5 — Limit what the small baseline shows

| # | Request | Where / verified | Decision | Action or justification |
|---|---|---|---|---|
| M5.1 | "Capacity is not the explanation" uses the proposed explanation as its own proof; the parameter-matched Neural SDE may simply be poorly optimized | Supplement l.181–187 | **FIX** | Take the reviewer's second route, with no new runs. Write: "(i) A parameter-matched Neural SDE does not reproduce the result under the same training budget", followed by "…consistent with, but not proof of, an advantage from the physics-based and connectome priors; a compressed Neural SDE may also be harder to optimize under the shared schedule." The main comparison with the full Neural SDE is unaffected. |
| M5.2 | "∼100× fewer parameters" | 220,176 in Table S6 is correct (M7.1). However, 2,215 counts each complex weight once; in real degrees of freedom it is 2,428 (ratio ≈ 91) | **FIX** | Keep "roughly two orders of magnitude" and drop "∼100×". Add a footnote to Table S6 on how complex parameters are counted. |

### M6 — Make the conclusions consistent with all metrics

| # | Request | Where / verified | Decision | Action or justification |
|---|---|---|---|---|
| M6.1 | Autocorrelation: the Hybrid is significantly worse, and the co-author's reply was wrong | **Verified**, Table S2: Hybrid 0.255 ± 0.024 vs. Coupled Hopf 0.142 ± 0.004\* and Neural SDE 0.161 ± 0.010\* (Holm) | **FIX** | Correct our reply to R7. Add one sentence to Results (fidelity) and one to the Discussion: of the three models, the Hybrid reproduces regional temporal memory worst, and its advantage lies in network structure and dynamics. Re-check the per-seed values after the M4.1 rerun. |
| M6.2 | The Supplement groups autocorrelation with "point-by-point" fidelity | Supplement l.75 | **FIX** | Separate the two. TS corr is point-by-point (and ill-posed across independent realizations); autocorrelation is a temporal statistic that is valid across non-aligned realizations. The same sentence should also say that the Hybrid is worst on autocorrelation. |
| M6.3 | TS PSD is defined but never reported | Methods l.104 (row 14 of the metrics table), l.193–196 | **FIX** (delete) | Remove PSD from the metrics table and the definitions; this shortens the paper. Alternative: add a column to S2 if the values are already in the per-seed JSONs. |
| M6.4 | "Identify the subject" overclaims | Discussion l.16 | **FIX** | Change to "carry subject-specific information". No identification-accuracy or permutation analysis: it would add content and is unnecessary once the wording is corrected. |
| M6.5 | No significant effect is not proof of no dynamic personalization | Discussion l.16 ("does not extend to"); Results l.127 is already careful | **FIX** | Change to "we detected no significant personalization of dynamic metrics". |
| M6.6 | Do not claim universal robustness or data efficiency | Results l.121 is already nuanced. The overclaims are in Discussion l.16 ("better data efficiency … demonstrates added value") and Intro l.27–28 ("fitted robustly to small samples", "robustness") | **FIX** | Align the Discussion and Intro with the Results paragraph: lowest mean error at both training-set sizes; not significant vs. the Neural SDE at 10 subjects; largest spread at 10 subjects. Keep the subsection heading (softening it is optional). |
| M6.7 | Time per epoch: the Hybrid is 4.5× slower than the Neural SDE | Table S6: 127.04 vs. 28.18 s/epoch | **CLARIFY** | Add one clause to the S6 text: fewer parameters does not mean faster fitting, because the per-node networks and the adjoint solver make each epoch slower. Check that no sentence implies the Hybrid is cheaper to train. |
| M6.8 | Name the Holm family; seeds are not independent cohorts; personalization is exploratory | Methods l.257 already says the refit is the unit of replication | **FIX** (wording) | Call the personalization analysis "exploratory" (Results and Supplement). Add one clause that seeds measure the reproducibility of the procedure within one cohort. Name the Holm family in the S2 caption. |

### M7 — Correct specifications and ensure reproducibility

| # | Request | Where / verified | Decision | Action or justification |
|---|---|---|---|---|
| M7.1 | Neural SDE parameters: the Methods text implies 93,904; Table S6 says 220,176 | **Verified: the table is right, the text is wrong** (finding 5) | **FIX** | Methods l.298: change "width 128" to 256 and "width 64" to 128. |
| M7.2 | The complexity regularizer R(Θ) is never defined | **Verified: unused** (`l2` weight = 0) | **FIX** | Remove R(Θ) from the objective (Methods l.245–247), or state that its weight is 0. |
| M7.3 | Coupled Hopf: 2 learned parameters (S6) vs. a grid over 3 (S11) | **Verified** (finding 6) | **CLARIFY** | Add one sentence to S11: the grid has one more degree of freedom (κ ∈ {0.1, 1}) than the gradient fit, so the comparison is conservative for gradient descent. |
| M7.4 | One exact configuration table per experiment | Most values are already in the Methods. Gaps: which local parameters are learned vs. fixed; the sign and normalization of C₀ (row-normalizing an FC matrix that has negative entries); noise normalization; rollout lengths and transient; how metrics are aggregated over windows and subjects | **OPTIONAL** (merge with M4.2), otherwise **CLARIFY** the gaps | If we add the M4.2 table, these become its columns. Otherwise add the missing facts inline in the Methods (about 3 sentences). |
| M7.5 | GNN-Hopf widths (found in this check; the reviewer did not flag it) | Supplement l.277 says "2 hidden layers, width 32"; the config uses 1 layer, width 16 | **FIX** | Correct the Supplement. |
| M7.6 | How the 0.05 s solver step aligns with 0.72 s observations | **Verified:** torchsde's fixed-step solver returns outputs at 0.72 s intervals by linearly interpolating between solver steps (0.72 / 0.05 = 14.4) | **CLARIFY** (+ OPTIONAL check) | Add one sentence to the Methods. Optional and cheap: re-evaluate the saved checkpoints with Δt = 0.025 s (evaluation only, no training) and report the result in one sentence. |
| M7.7 | Identify the code version that produced the tables | The Zenodo DOI is already a TODO in `main.tex` | **FIX** (already planned) | Mint the DOI on the release tag after the reruns and cite it in the data-availability paragraph. No extra text. |

### M8 — Small errors that raise avoidable doubts

| # | Request | Where / verified | Decision | Action or justification |
|---|---|---|---|---|
| M8.1 | "the hybrid generative models achieves" | `main.tex` l.143 (abstract) | **FIX** | Change to "the hybrid generative model achieves". |
| M8.2 | "only partial representing" | Not present in the current sources | **KEEP** | Already removed. |
| M8.3 | Missing space before "No physical materials" | That sentence is commented out (`main.tex` l.182) and does not appear in the PDF | **KEEP** | Nothing to do; the reviewer probably read an older PDF. |
| M8.4 | phFC: a mean cosine near 1 means a phase difference near 0 mod 2π, not just any constant difference | Methods l.165 | **FIX** | Change to "close to 1 indicates the two regions stay nearly in phase; a constant non-zero lag δ gives cos δ". |
| M8.5 | Centering a recording does not imply zero expectation at every time point | Methods l.60 | **FIX** | Change to "we assume E z(t) ≈ 0, motivated by the per-recording centering". |
| M8.6 | Entropy and LZW are not a clinical biomarker; their values depend on discretization and vectorization order | Results l.136, Supplement | **FIX** (wording) | The current text makes no clinical claim. Add "exploratory" and one clause that the values depend on binning and vectorization order. Change "We demonstrate" to "We observe". |
| M8.7 | The "More interpretable" arrow in Figure 3 (architectures) | Figure caption, Results l.53 | **FIX** (caption only) | Add "schematic; the interpretability axis is qualitative". No redraw. |

---

## C. Order of work

1. **Decide on M4.1 (rerun or disclose).** If we rerun: patch [train_models.py](examples/train_models.py) so the training cohort is always used, relaunch, then re-aggregate Tables 1, S2, S4 and S5 and Fig. 3d. Every edit that depends on numbers (M1.4, M5.2, M6.1, M6.6) waits for this.
2. **Text-only FIX items** (independent of the rerun): M2.1–M2.3, M3.1, M4.3–M4.5, M5.1, M6.2–M6.5, M6.8, M7.1–M7.3, M7.5, and M8.*. Most are one-sentence edits.
3. **CLARIFY items that need information:** M4.7 (HCP subset details, from co-authors) and M4.6 (provenance of the Fig. 3d 65-subject runs, from the cluster).
4. **Optional items to decide together:** M3.2 (control row), M4.2/M7.4 (protocol table), M7.6 (step-size check).
5. Rebuild, check that the abstract is ≤ 150 words, and mint the Zenodo DOI.

**Net effect on length** without any optional items: about 8–10 sentences of protocol clarification, minus one metric definition (PSD). The paper stays essentially the same size.

---

## Appendix: original review (verbatim)

### Reconsideración de los ocho comentarios del coautor

El informe anterior contiene ocho comentarios de Word. La decisión siguiente tiene en cuenta tanto esas respuestas como lo que efectivamente aparece en la nueva versión. Los números corresponden a los identificadores de los comentarios [R].

| Comentario | Respuesta del coautor | Decisión de esta evaluación |
|---|---|---|
| 0 Predicción independiente | No realizado porque exigiría datos nuevos. | Aplazable. Delimitar el artículo como método de reproducción estadística; mantener prospectivas las intervenciones. |
| 1 Interpretabilidad fisiológica | Realizado; se discute reproducibilidad y no se afirma interpretabilidad completa. | Parcial. Reproducibilidad de métricas no demuestra estabilidad ni identificación fisiológica del acoplamiento. |
| 2 Validación externa | No realizada; se separan entrenamiento y test. | Aplazable. El split interno no equivale a validación externa; explicitar una sola cohorte sana. |
| 3 Contribución central | Realizado; se refinaron introducción, resumen y discusión. | Sustancialmente atendido. Reforzar la síntesis cuantitativa tras verificar parámetros y afirmaciones. |
| 4 Personalización estática y dinámica | No realizado; las tablas y los resultados ya lo muestran. | Resuelto en lo esencial. Páginas 11 y 13 ya lo explicitan; no repetirlo como ausencia del manuscrito. |
| 5 Alineación del surrogate | No realizado. | Pendiente importante. La cota MSE–KS no resuelve el emparejamiento entre realizaciones estocásticas. |
| 6 Baseline neuronal pequeña | No realizado; se considera clara la explicación mediante el prior. | Reconsiderar. Comprobar optimización o limitar la atribución causal al presupuesto y arquitectura ensayados. |
| 7 Autocorrelación | Realizado; se afirma que la diferencia no es significativa. | Debe corregirse. La Tabla S2 sí marca diferencias significativas; reconciliar texto, comentario y análisis. |

Acepto, por tanto, no exigir una nueva cohorte ni una predicción clínica para este envío. También doy por atendida la distinción entre personalización estática y dinámica. No considero cerradas las objeciones de surrogate ni de equidad de la baseline pequeña. La discrepancia de autocorrelación requiere una corrección concreta, y la interpretabilidad puede delimitarse correctamente sin emprender nuevos experimentos fisiológicos.

La comparación previa con ProteinTalks debe servir para distinguir demostración metodológica y validación de aplicaciones. No debe convertirse en la exigencia de imitar la escala experimental de un trabajo de proteómica. Tampoco justifica inferir superioridad matemática de un artículo a partir de la revista en que aparece [R, apartados 3 y 6].

### 5 Cambios imprescindibles antes del envío

Los ocho bloques siguientes separan correcciones verificables, aclaraciones del protocolo y cambios que dependen de una comprobación. El cierre de cada bloque se define para evitar una revisión indefinida.

#### M1 Formular exactamente lo que se aprende

**Dónde.** Título, resumen, final de la introducción, Methods p. 14 y Discussion pp. 12–13.

**Acción.** Formular el problema como ajuste generativo de descriptores seleccionados, no como recuperación demostrada de la verdadera dinámica cerebral. Presentar la contribución integrada frente a los cuatro antecedentes de la sección 3. Sustituir "fully interpretable" por una descripción verificable del término Hopf explícito y de la estructura de grafo.

El resumen debe incluir un resultado numérico, no sólo calificativos: 94 sujetos, mejora del compuesto respecto de los dos comparadores y carácter no uniforme de esa mejora. La contabilidad de parámetros puede añadirse una vez corregida M7. No es necesario prometer capacidades clínicas para dar relevancia al resultado.

**Criterio de cierre.** Un lector debe poder identificar, antes de Results, el observable modelado, la información disponible, los componentes aprendidos, el criterio de éxito y la diferencia con el antecedente más cercano.

#### M2 Corregir el alcance de la restricción de acoplamiento

**Dónde.** Afirmaciones sobre aprender "only" el acoplamiento, pp. 3–4, 7 y 13; ecuación de Hybrid Hopf, p. 26; comparación con GNN Hopf, S9–S10.

La red de salida recibe el mensaje agregado y el estado del nodo. Aunque el mensaje sea cero, puede producir una corrección dependiente del estado local. Para garantizar que no modifica la dinámica desacoplada se necesita una condición como la siguiente:

$$\psi_{\text{out}}(0, z_i) = 0.$$

**Acción mínima.** Comprobar el código efectivamente usado. Si esa condición está impuesta, explicarla. Si no lo está, decir que el modelo conserva un término Hopf explícito y aprende interacciones condicionadas por el grafo. No incorporar retrospectivamente a la ecuación una restricción que los experimentos no utilizaron. Añadirla sería un cambio de modelo y requeriría reevaluación.

Debe corregirse también la comparación suplementaria que atribuye al GNN el uso del estado receptor como diferencia frente al Hybrid: la ecuación escrita de Hybrid ya utiliza ese estado. Las diferencias reales incluyen el encoder y la parametrización de rango bajo frente a una matriz completa.

**Interpretación.** La matriz aprendida no es directamente una matriz de influencia fisiológica identificada: el efecto de un nodo sobre otro depende también de las derivadas de las redes de entrada y salida. La factorización y las reparametrizaciones introducen no unicidad. Mantener "componente explícito" o "prior estructurado" es defendible; "mechanistic interpretability" necesita una delimitación expresa.

**Criterio de cierre.** Las afirmaciones verbales deben coincidir con la ecuación y con la implementación que produjo las tablas. Este punto puede cerrarse mediante precisión conceptual, sin cambiar la arquitectura.

#### M3 Resolver la objeción sobre la pérdida sustituta

**Dónde.** Pp. 6, 17–18 y S1–S2. Este es el principal punto metodológico aún abierto del informe anterior.

**Primera cuestión.** La proposición general es correcta:

$$D_{KS} \le \frac{L_{MSE}}{r^2} + Q_b(r), \qquad r > 0.$$

Pero una distribución empírica finita tiene átomos. Su función de concentración no puede quedar acotada por una constante por r hasta escalas arbitrariamente pequeñas: conserva al menos la masa de su átomo mayor. El corolario ya menciona un rango de escalas, pero debe exigir que el radio óptimo pertenezca a ese rango. La conclusión posterior sobre control con exponente un tercio no debe presentarse sin esta condición. La solución más breve es mantener la cota exacta con la función de concentración y formular el corolario como estrictamente condicional.

**Segunda cuestión.** La cota no demuestra que emparejar la misma entrada temporal en dos matrices FCD independientes sea un objetivo estadísticamente bien calibrado. Para matrices aleatorias independientes, la esperanza del error cuadrático contiene tanto la diferencia entre sus medias como la variabilidad de ambas. Reducir la variabilidad simulada puede reducir el error aun cuando no mejore la distribución. Es el mismo tipo de dificultad que el artículo explica acertadamente al rechazar el ajuste punto a punto de trayectorias estocásticas. La condición inicial compartida requiere matizar la independencia, pero no elimina la objeción después de perderse esa dependencia.

**Control acotado recomendado antes de este tercer envío.** Mantener arquitectura, particiones y presupuesto de ajuste, y comparar la pérdida por entradas con una pérdida de distribución sobre esas mismas entradas, por ejemplo MMD o distancia cuadrática entre entradas ordenadas. Con muestras de igual tamaño, esta última corresponde al transporte cuadrático unidimensional empírico y admite gradientes casi en todas partes. Bastan unas pocas semillas ya usadas para un control exploratorio, reportando KS de FCD y phFCD, autocorrelación y variabilidad entre nuevas realizaciones. No presentarlo como una nueva prueba confirmatoria de gran potencia.

Una permutación conjunta de filas y columnas de la matriz empírica permite además mostrar, sin reentrenar, que se conserva la distribución de entradas mientras cambia el MSE por posiciones. Es un diagnóstico de la objeción, no un sustituto del control de entrenamiento. Si se decide no añadir este control, la pérdida debe describirse expresamente como un surrogate heurístico útil y su limitación debe discutirse; esa opción deja más riesgo de objeción del referee.

**Criterio de cierre.** Corolario matemáticamente preciso y una justificación empírica proporcionada del objetivo, o una delimitación explícita y honesta de lo que la cota no demuestra. La respuesta "Not done" no basta para dar el asunto por resuelto.

#### M4 Hacer inequívoca la separación de información

**Dónde.** Data and preprocessing p. 15, Training pp. 21–23, experimento con diez sujetos pp. 9–10 y personalización S4–S5.

**Acción.** Añadir un esquema o una tabla de los experimentos que indique sujetos, segmentos temporales, fuente del prior funcional, fuente de las frecuencias, selección de checkpoint y conjunto usado para evaluación. Diferenciar el experimento principal con partición por semilla del experimento de tamaño muestral que mantiene los mismos quince sujetos de test. Corregir la referencia suplementaria a ajustes con N = 94 si el entrenamiento poblacional corresponde realmente a 65 sujetos.

**Comprobación del preprocesamiento.** Aclarar si estandarización, filtro y Hilbert se calculan antes o después de dividir cada registro. Los filtros no causales y la transformada de Hilbert aplicada a la señal completa pueden utilizar muestras que después se presentan como reservadas. El PDF no permite establecer la magnitud de ese posible efecto. Si el código cruza las fronteras temporales, repetir la evaluación afectada con segmentos separados o con un procedimiento y márgenes de exclusión que eviten compartir información.

**Información poblacional.** S4 mantiene un prior de conectoma común para los ajustes individuales. El resultado debe describirse como calibración individual condicionada a un prior poblacional. Especificar qué sujetos y segmentos lo construyen. En el experimento de diez sujetos, indicar si el prior se reestima con esos diez o si utiliza la cohorte mayor: ambas opciones son posibles, pero responden a preguntas diferentes sobre eficiencia de datos.

Documentar la selección del subconjunto HCP y si los sujetos son no emparentados; si incluye familias, usar una separación que respete esos grupos para la afirmación de generalización. No presupongo que haya una fuga efectiva: la tarea es establecer el protocolo. Identificar la sesión, dirección de adquisición y criterios por los que se eligieron los 94 registros.

**Criterio de cierre.** El lector debe poder reconstruir qué información estaba disponible en cada ajuste y comprobar que las etiquetas de "held out", "ten subjects" y "single subject" describen exactamente ese uso de datos.

#### M5 Delimitar lo que demuestra la baseline pequeña

**Dónde.** Tabla S4 y su interpretación en S7; afirmaciones de que la capacidad no explica el resultado, pp. 8–9 y 13.

El Neural SDE de 2.806 parámetros obtiene FC corr de −0,007, frente a 0,916 del Hybrid de referencia en esa ablación. El contraste es grande, pero un modelo neuronal extremadamente comprimido, entrenado con la misma agenda y tasa que otro, puede sufrir un cuello de botella o una dificultad de optimización. Igualar el número de parámetros no iguala esas propiedades.

**Dos formas proporcionadas de cerrar el punto.** Si se desea conservar la atribución fuerte al prior, revisar configuración, curvas de entrenamiento y validación, y una pequeña variación razonable de tasa de aprendizaje o presupuesto de entrenamiento elegida con validación, no con test. Si no se realizará esa comprobación, mantener la fila como resultado del modelo ensayado y sustituir "Capacity is not the explanation" por una conclusión condicionada a esa arquitectura y presupuesto.

La respuesta del coautor no resuelve esta cuestión porque utiliza la explicación propuesta como prueba de sí misma. No hace falta incorporar una batería extensa de modelos nuevos. La comparación principal con la Neural SDE grande conserva interés aunque la baseline comprimida se presente con cautela.

**Criterio de cierre.** Que el lector pueda distinguir evidencia a favor de un sesgo estructural útil de una demostración de que se han excluido otras explicaciones.

#### M6 Reconciliar las conclusiones con todas las métricas

**Dónde.** Tabla 1, Discussion, Tabla S2 y comentario 7 del Word.

**Autocorrelación.** La Tabla S2 informa 0,255 ± 0,024 para Hybrid Hopf, frente a 0,142 ± 0,004 para Coupled Hopf y 0,161 ± 0,010 para Neural SDE. Las dos comparaciones llevan asterisco de significación corregida frente al Hybrid. Esto contradice la afirmación del coautor de que la diferencia no es significativa. Revisar los valores por semilla, los p-valores y la leyenda; corregir lo que corresponda. Si la tabla es correcta, reconocer en la discusión principal la peor reproducción de memoria temporal regional.

La autocorrelación es una estadística temporal válida aunque dos realizaciones no estén alineadas. No debe confundirse con correlación punto a punto entre trayectorias independientes. El párrafo de S3 que reúne ambas como fidelidad punto a punto debe reformularse. La PSD está definida como diagnóstico de evaluación, pero sus valores no figuran en las tablas comparativas: informar el resultado si se calculó o precisar el alcance real de lo reportado.

**Personalización.** La limitación a estructura estática ya está bien explicada. Queda una precisión distinta: superar la media de nueve modelos ajenos no implica identificar correctamente al sujeto. Sustituir "identify the subject" de p. 13 por "carry subject-specific information", o calcular rango y exactitud de identificación a partir de la matriz de resultados ya disponible. No interpretar falta de significación dinámica como demostración de ausencia de cualquier efecto.

**Robustez y coste.** A diez sujetos, el Hybrid tiene menor error medio, pero mayor dispersión y ventaja no significativa frente a Neural SDE. No afirmar superioridad universal en robustez o eficiencia muestral. La Tabla S6 da 127,04 s por época para Hybrid y 28,18 s para Neural SDE: unas 4,5 veces más tiempo por época en esa implementación. Menos parámetros y viabilidad en alta dimensión no equivalen a menor tiempo de ajuste.

**Estadística.** Mantener explícito qué familia de comparaciones recibe corrección Holm. Los contrastes por semilla describen reproducibilidad del procedimiento en esta cohorte; no son nueve cohortes biológicas independientes. La personalización con diez objetivos debe calificarse de exploratoria. Si se mantiene una afirmación de identificación, una prueba por permutación de etiquetas en la matriz de ajustes y objetivos sería más directa que inferirla del contraste propio frente a media ajena.

**Criterio de cierre.** Resumen, resultados y discusión deben comunicar la misma ventaja específica y los mismos compromisos, sin apoyarse selectivamente en el compuesto.

#### M7 Corregir especificaciones y garantizar reproducción

**Dónde.** Methods pp. 22–26 y Tablas S5–S6. Hay una inconsistencia cuantitativa verificable a partir de la arquitectura escrita.

| Componente descrito | Parámetros con capas densas y bias |
|---|---|
| Drift real de 200 a 128 a 128 a 200 | 68.040 |
| Difusión real de 200 a 64 a 200 | 25.864 |
| Total deducido de Methods | 93.904 |
| Total declarado en Tabla S6 | 220.176 |

El total de 220.176 coincide, bajo esa misma convención, con anchuras 256 para las dos capas ocultas del drift y 128 para la difusión. Verificar los archivos de configuración guardados y corregir el texto o la tabla. El cociente entre 220.176 y 2.215 es aproximadamente 99; entre 93.904 y 2.215 sería aproximadamente 42. No destacar una reducción de cien veces sin esta comprobación.

Preparar una única tabla de configuración exacta por experimento. Debe indicar qué parámetros locales se aprenden y cuáles se fijan; tamaños de redes y rango; tratamiento de pesos negativos y normalización de la matriz funcional; normalización del ruido complejo; duración de los rollouts de evaluación y tratamiento del transitorio; pesos y normalizaciones de pérdidas. Definir el regularizador de complejidad y su coeficiente de p. 22, o eliminarlo si no se utilizó.

La Tabla S6 declara dos parámetros aprendidos para Coupled Hopf, mientras S11 describe grid search sobre tres. Para atribuir diferencias al optimizador, la comparación debe usar los mismos grados de libertad y objetivo, o explicar claramente la diferencia. Indicar además cómo se agregan métricas entre ventanas y sujetos, cómo se reproduce la prueba estadística y cuál es la referencia de código que produjo exactamente las tablas.

La definición de un paso interno de 0,05 s junto a observaciones cada 0,72 s requiere explicar la alineación o interpolación del solver. Como control numérico acotado, si no existe ya, conviene verificar en checkpoints guardados que reducir el paso no altera materialmente las métricas ni produce inestabilidad. No hace falta introducir una teoría nueva de integración estocástica.

**Criterio de cierre.** Un revisor debe poder reconstruir las arquitecturas y reproducir las tablas desde una versión identificable del código, sus configuraciones y resultados por semilla. La disponibilidad de un repositorio no reemplaza la concordancia de esas especificaciones.

#### M8 Reparar las afirmaciones menores que crean dudas evitables

**Dónde.** Resumen, definiciones de métricas y descripciones de figuras.

Corregir "the hybrid generative models achieves" y "only partial representing", así como el espacio ausente antes de "No physical materials" en la disponibilidad de código.

En p. 19, un promedio de coseno próximo a uno implica diferencias de fase próximas a cero módulo dos pi; no cualquier diferencia constante. Una fase constante distinta de cero puede producir otro valor. Precisar la explicación de phFC.

En p. 16, el centrado temporal de un registro no implica por sí solo esperanza probabilística nula en cada instante. Formular la hipótesis de centrado y decorrelación como aproximación o supuesto del argumento.

Separar los indicadores exploratorios de entropía y LZW de una validación clínica. Su cálculo sobre entradas vectorizadas de FCD no demuestra automáticamente un biomarcador de conciencia; el resultado depende de discretización y orden de vectorización.

Alinear la Figura 3 con las ecuaciones: explicitar las simplificaciones esquemáticas de parámetros y evitar que la flecha de "More interpretable" parezca una medida de interpretabilidad validada.

**Criterio de cierre.** Una última lectura de precisión debe eliminar errores que un especialista detectaría rápidamente sin cambiar el argumento central.
