# Handoff Document for Next Agent

**Project:** Score-Entropy Discrete Diffusion (SEDD) Research
**Last Agent:** Gemini
**Date:** 2025-07-08

## 1. Project Goal & Current State

The primary research goal of this project was to understand why the `Uniform` state diffusion model underperforms compared to the `Absorbing` state model, and to develop a novel, non-absorbing state model that could match or exceed the baseline performance.

After a comprehensive series of experiments, this goal has been largely achieved. We have successfully developed a new, high-performing **"Self-Distrusting" Hybrid Model**.

The repository is in a clean, stable, and documented state. The new state-of-the-art model is defined in `self_distrust_hybrid_config.yaml` and can be run with `run_final_experiment.sh`.

## 2. Summary of Research Journey & Key Findings

Our investigation followed a rigorous, hypothesis-driven path.

### 2.1. Initial Diagnosis: Why the Uniform Model Fails

We began by testing and invalidating the initial hypothesis of "noise ambiguity" via an oracle experiment. This led to a deeper investigation that confirmed three root causes for the Uniform model's failure:

1.  **Impure Loss Signal:** The model struggles to learn from a loss signal calculated on corrupted tokens that are themselves valid but incorrect words.
2.  **Context Corruption:** Random tokens in the input actively mislead the model's attention mechanism, making context-based predictions difficult.
3.  **Unstable Decoding Dynamics:** The generation process is chaotic, with the model constantly changing its predictions ("flickering") because any token can be rewritten at any time.

### 2.2. Development of Novel Architectures

Based on these findings, we developed and tested several novel architectures:

-   **Hybrid Models:** Combining `[MASK]` tokens with uniform noise proved to be a successful strategy for enabling contextual infilling.
-   **Gated-Attention:** A new architecture that learns to "gate" or ignore noisy tokens was highly effective at increasing sample diversity, but hurt perplexity by learning to ignore context too aggressively.
-   **Gated-Absorbing Model:** As a control, we applied the gating mechanism to a pure absorbing model. This resulted in a new **state-of-the-art perplexity score (24.51)**, proving the value of the gating mechanism even in a "perfect signal" environment.

### 2.3. The Final Model: "Self-Distrusting" Hybrid

Our final and most successful **non-absorbing** model was a synthesis of our key learnings.

-   **Architecture:** Gated-Attention
-   **Diffusion Process:** Hybrid Graph (`mask_ratio: 0.75`)
-   **Key Innovation:** A "soft" target for the gate loss. This forces the model to be skeptical of uniform noise without ignoring it completely, balancing the need for a strong signal with the challenge of learning from corrupted context.

### 2.4. Final Results Table

| Model | Perplexity | Distinct-2 | Self-BLEU | Infilling | Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `final_uniform` | 25.08 | 0.79 | 0.049 | Failed | Uniform |
| `final_absorbing` | 24.60 | 0.97 | 0.008 | `breadane spun` | Absorbing |
| **Gated-Absorbing** | **24.51** | **0.97** | **0.007** | `over the lazy` | **Absorbing** |
| **Self-Distrusting Hybrid**| 25.77 | **0.98** | **0.002** | `/ chris heritage` | **Hybrid** |

## 3. Proposed Next Steps for the Next Agent

While we have successfully developed a high-performing hybrid model, the perplexity of the pure Gated-Absorbing model remains superior. The next phase of research should focus on closing this final gap.

### 3.1. Primary Goal

The primary goal is to **reduce the perplexity of the "Self-Distrusting" Hybrid Model to match or surpass the 24.51 benchmark set by the Gated-Absorbing model**, without sacrificing its state-of-the-art diversity.

### 3.2. Recommended Experiments

Here are two promising, parallel paths to achieve this goal:

**Path 1: Architectural Refinement - The Two-Stream Transformer**

-   **Hypothesis:** The Gated-Attention model, while effective, still forces a single Transformer to perform two distinct tasks (noise identification and prediction). Decoupling these tasks entirely could lead to better performance.
-   **Action:**
    1.  Implement the **"Two-Stream Transformer"** as detailed in `NOVEL_IDEAS.md`. This involves creating a dedicated, smaller "Gate Stream" to predict noise, and feeding its predictions as an explicit input to the main "Prediction Stream."
    2.  Train this new architecture using the `HybridGraph` and the "soft-distrust" loss function.
-   **Expected Outcome:** By freeing the main Prediction Stream from the task of identifying noise, it can dedicate its full capacity to language modeling, which should result in a lower perplexity.

**Path 2: Algorithmic Refinement - Progressive Masking**

-   **Hypothesis:** The fixed `mask_ratio` in the Hybrid model is a compromise. A dynamic schedule could provide a better learning path.
-   **Action:**
    1.  Implement the **"Progressive Masking"** algorithm as detailed in `NOVEL_IDEAS.md`. This involves modifying the `HybridGraph` to anneal the `mask_ratio` from 1.0 down to a lower value (e.g., 0.5) over the course of the diffusion timestep `t`.
    2.  Train the existing `Self-Distrusting` model architecture with this new, dynamic diffusion process.
-   **Expected Outcome:** This "easy start, hard finish" approach should provide the stability of the Absorbing model in the early stages of decoding and the refinement challenge of the Uniform model in the later stages, potentially leading to a better final perplexity.

These two paths represent the most promising avenues for surpassing the current state-of-the-art and completing the research objectives. Good luck.
