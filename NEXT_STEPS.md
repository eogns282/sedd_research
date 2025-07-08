# Project Handoff & Next Steps: Improving the Uniform Diffusion Model

## 1. Project Overview & Current State

This repository contains a robust, verified, and well-documented research framework for comparing **Uniform** and **Absorbing** state discrete diffusion models. The codebase is stable and all previous bugs have been resolved.

## 2. Summary of Key Findings & The Core Research Problem

Our initial experiments have definitively shown that the **Absorbing state model significantly outperforms the Uniform state model**, particularly in quantitative perplexity and qualitative contextual tasks like infilling.

This leads to our central research question: **Why does the Uniform model struggle, and how can we improve it?**

**Our primary hypothesis was that the Uniform model's main difficulty is *noise ambiguity*.** In the Absorbing model, the `[MASK]` token provides a clear, unambiguous signal of which parts of the input are noise. In the Uniform model, any token could be either original data or random noise, making the learning problem for the denoising model much more difficult.

## 3. The "Oracle" Experiment: A Surprising Result

To test our hypothesis, we conducted an "oracle" experiment where we gave the Uniform model perfect information about which tokens were noise during training. The results were unexpected:

-   **Perplexity:** The `UniformOracle` model's perplexity was not significantly better than the standard `Uniform` model, and was still much worse than the `Absorbing` model.
-   **Diversity:** The oracle model *did* show a significant improvement in sample diversity.
-   **Infilling:** The oracle model completely failed at infilling, just like the standard `Uniform` model.

**Conclusion:** This experiment strongly suggests that **noise ambiguity is NOT the primary reason for the Uniform model's failure.** The core problem seems to be more fundamental to the uniform noising process itself. Even when the model knows exactly which tokens to fix, it cannot learn to do so effectively in a contextual manner.

## 4. New Research Direction: Hybrid Diffusion

The failure of the oracle experiment means that our original plan (Step 3: Predictive Model) is no longer viable. A new approach is needed.

Our new hypothesis is that the **Absorbing model's strength comes from its explicit `[MASK]` token**, which provides a strong, structural signal for the model to learn from. The Uniform model's weakness is the lack of such a signal.

Therefore, the new research goal is to **develop a hybrid diffusion model** that combines the uniform noising strategy with an explicit mask signal.

### Step 1 (Immediate Task): Implement a `HybridGraph`

-   **Action:** Create a new graph type, `HybridGraph`, in `src/diffusion/graph.py`.
-   **Implementation Details:**
    -   The `HybridGraph`'s `sample_transition` method will take a new parameter, `mask_ratio` (e.g., 0.5).
    -   When adding noise, it will randomly select a portion of the tokens to corrupt based on the `corruption_prob`.
    -   Of the selected tokens, it will apply **absorbing noise** (replace with `[MASK]`) to a fraction of them determined by `mask_ratio`, and **uniform noise** (replace with a random token) to the rest.
    -   This will create a noisy sample that contains a mix of original tokens, `[MASK]` tokens, and random noise tokens.

### Step 2: Train and Analyze the Hybrid Model

-   **Action:** Create a `hybrid_config.yaml` and a `run_hybrid_experiment.sh` script.
-   **Analysis:** Train the `HybridGraph` model with different `mask_ratio` values (e.g., 0.25, 0.5, 0.75) and compare the results to the `final_uniform` and `final_absorbing` baselines. This will allow us to see how the ratio of absorbing to uniform noise affects performance.

This new direction is a direct response to the experimental evidence from the oracle experiment and provides a clear path forward for improving the Uniform diffusion model.
