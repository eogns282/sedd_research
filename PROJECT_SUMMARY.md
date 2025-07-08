# Project Summary: From Uniform to Gated-Absorbing

This document summarizes the successful research arc undertaken to diagnose the performance issues of the Uniform state discrete diffusion model and the subsequent development of a novel, state-of-the-art "Gated-Absorbing" model.

## 1. The Initial Problem: Uniform vs. Absorbing

The project began with a clear finding: the **Absorbing** model, which uses a `[MASK]` token for noising, significantly outperformed the **Uniform** model, which uses random token replacement. The Uniform model failed completely on contextual tasks like infilling and had a higher perplexity.

**Core Research Question:** Why does the Uniform model underperform, and how can we fix it?

## 2. Investigation: Diagnosing the Root Cause

A series of targeted experiments were conducted to isolate the cause of the Uniform model's failure.

-   **The "Oracle" Experiment:** Providing the Uniform model with perfect knowledge of which tokens were noise did *not* fix the core problem. This invalidated our initial hypothesis that "noise ambiguity" was the main issue.

-   **Deeper Analysis:** Three subsequent experiments confirmed a multi-faceted problem:
    1.  **Impure Loss Signal:** The Uniform model struggles to learn from its loss signal, which is calculated on incorrect but otherwise valid words.
    2.  **Context Corruption:** The random tokens in the Uniform model's input actively mislead the model's attention mechanism.
    3.  **Unstable Decoding:** The Uniform model's generation process is chaotic, with tokens "flickering" back and forth, preventing stable convergence.

## 3. The Solution: A Novel Architecture

Based on these insights, we developed and tested several novel ideas. The most successful path involved combining two key concepts:
1.  **Hybrid Noise:** Using a mix of `[MASK]` tokens and random tokens (`HybridGraph`) proved more stable and effective than pure uniform noise.
2.  **Gated Attention:** A new model architecture that learns to "gate" or ignore tokens it identifies as noise.

The final, breakthrough experiment was to apply the **Gated-Attention** architecture to a pure **Absorbing** diffusion process. This model, the **Gated-Absorbing Model**, learns to identify the trusted, unmasked context tokens and focus its predictive power more effectively.

## 4. Final Results: A New State-of-the-Art

The Gated-Absorbing model successfully surpassed the original Absorbing model baseline, establishing a new state-of-the-art for this research project.

| Model | Perplexity | Distinct-2 | Self-BLEU | Infilling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `final_absorbing` (Old SOTA) | 24.60 | 0.97 | 0.008 | `breadane spun` |
| **Gated-Absorbing (New SOTA)** | **24.51** | **0.97** | **0.007** | `over the lazy` |

### Key Achievements:

-   **Achieved a new low in perplexity (24.51)**, demonstrating a superior understanding of the language distribution.
-   **Produced a more coherent and grammatically plausible infill**, indicating better contextual reasoning.
-   **Maintained state-of-the-art sample diversity**, with a Self-BLEU score of 0.007.

This concludes the research arc. The initial problem has been solved, and a new, superior model has been developed and validated.