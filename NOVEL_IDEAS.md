# Novel Ideas for Improving the Uniform Diffusion Model

Based on a series of targeted experiments, we have confirmed three core hypotheses for why the Uniform diffusion model underperforms relative to the Absorbing model:

1.  **Impure Loss Signal:** The Uniform model is trained on a "confused" objective, where the loss signal from corrupted tokens is harder to learn from than the clean "fill-in-the-blank" task of the Absorbing model.
2.  **Context Corruption:** Random tokens in the Uniform model's input actively provide misleading contextual information, which the model must learn to ignore.
3.  **Unstable Decoding Dynamics:** The Uniform model's ability to change any token at any time leads to a chaotic "flickering" during generation, preventing stable convergence to a coherent sequence.

This document proposes three novel, code-based trial ideas to directly address these diagnosed issues.

---

## Idea 1: The Simple Tweak - "Dual-Channel Loss"

### 1.1. Core Concept

This idea addresses the "Impure Loss Signal" problem by explicitly separating the loss calculation for tokens that were part of the original context versus tokens that were corrupted. By applying a higher weight to the loss from corrupted tokens, we can force the model to prioritize learning the primary denoising task, effectively amplifying the "correct" learning signal.

### 1.2. Problem Addressed

-   **Primary:** Hypothesis 1 (Impure Loss Signal).
-   **Secondary:** Hypothesis 2 (Context Corruption). By focusing the model more on the corrupted tokens, it may implicitly learn to weight their misleading context less.

### 1.3. Detailed Implementation Plan

This is a minimal-change, high-impact proposal that only requires modifying the loss function.

1.  **Modify `uniform_config.yaml`:**
    -   Add a new hyperparameter under the `training` section:
        ```yaml
        training:
          # ... existing parameters
          loss_corrupted_weight: 1.5 # (or 2.0, etc.)
        ```

2.  **Modify `src/losses.py`:**
    -   The `loss_fn` function already receives the `corruption_mask` from the `diffusion_process`. We will use it to apply the weights.
    -   The implementation would look like this:

        ```python
        # Inside loss_fn, after calculating loss_per_token
        
        # Check if the weighting is enabled in the config
        loss_weight = getattr(config.training, 'loss_corrupted_weight', 1.0)

        if loss_weight > 1.0 and corruption_mask is not None:
            # Create a tensor of weights, default to 1.0
            weights = torch.ones_like(loss_per_token)
            
            # Apply the higher weight to the positions that were corrupted
            weights[corruption_mask] = loss_weight
            
            # Calculate the final loss as the weighted mean
            final_loss = (loss_per_token * weights).mean()
        else:
            # Default behavior if no weight is specified
            final_loss = loss_per_token.mean()

        # Return the final loss for backpropagation
        return final_loss
        ```

### 1.4. Expected Outcome & Metrics

-   **Prediction:** The model's training will be more efficient. The higher penalty for incorrect predictions on noised tokens should lead to faster convergence and a lower overall validation loss.
-   **Success Metrics:**
    -   **Primary:** A significant reduction in the final `perplexity` score compared to the vanilla `final_uniform` model.
    -   **Secondary:** Potential improvement in the `infilling` task, as the model becomes more adept at correcting noise.

### 1.5. Pros & Cons

-   **Pros:**
    -   Extremely simple to implement; requires changing only a few lines of code.
    -   Low risk; it's a simple hyperparameter tweak on the existing architecture.
    -   Directly targets the empirically validated "impure signal" problem.
-   **Cons:**
    -   It's a "brute force" method and may not be the most elegant solution.
    -   It might not fully solve the context corruption issue, as the misleading tokens are still present in the attention mechanism.

---

## Idea 2: The Architectural Change - "Gated Noise-Attention"

### 2.1. Core Concept

This idea addresses the "Context Corruption" problem by fundamentally changing the Transformer architecture. We will add a mechanism that allows the model to learn to dynamically "gate" or ignore tokens it believes are noise. This prevents misleading context from polluting the attention calculations for other tokens.

### 2.2. Problem Addressed

-   **Primary:** Hypothesis 2 (Context Corruption).
-   **Secondary:** Hypothesis 3 (Unstable Decoding). By learning to ignore noise, the model's own (potentially noisy) predictions in later steps will have less of a corrupting influence, leading to more stable generation.

### 2.3. Detailed Implementation Plan

This is a more involved change that requires modifying the model's core architecture.

1.  **Modify `src/model.py` - `TransformerModel`:**
    -   In the `__init__` method, add a new linear layer that will act as the gate prediction head.
        ```python
        self.gate_predictor = nn.Linear(self.d_model, 1)
        ```
    -   In the `forward` pass, after the main transformer encoder runs, pass its output through the gate predictor and a sigmoid function to get a "trustworthiness" score for each token.
        ```python
        # After transformer_output = self.transformer_encoder(x)
        gate_scores = torch.sigmoid(self.gate_predictor(transformer_output)) # Shape: [batch, seq_len, 1]
        ```
    -   The key innovation is to **re-run the final layer of the transformer** (or the entire transformer, for a more powerful effect) but with the gate scores applied. For simplicity, we can apply it before the final output layer.
        ```python
        # Apply the gate before the final output layer
        gated_output = transformer_output * gate_scores
        logits = self.output_layer(gated_output)
        ```
    -   **Crucially, this gate prediction needs to be trained.** We need a loss for it. We can use the `corruption_mask` from the noising process as the ground truth label for the gate.
        -   Modify `losses.py` to calculate a `gate_loss` (e.g., Binary Cross-Entropy) between the predicted `gate_scores` and the `corruption_mask`.
        -   The total loss returned would be `total_loss = cross_entropy_loss + gate_loss_weight * gate_loss`.

### 2.4. Expected Outcome & Metrics

-   **Prediction:** This should dramatically improve the model's ability to handle corrupted context. The model will learn to identify and effectively ignore noise tokens.
-   **Success Metrics:**
    -   **Primary:** A massive improvement in the `infilling` task. The model should be able to correctly fill in blanks even when surrounded by misleading random tokens.
    -   **Secondary:** A large reduction in `perplexity`, likely surpassing the simple weighted loss scheme.
    -   **Tertiary:** We could even analyze the learned `gate_scores` themselves as a new metric to see if the model is correctly identifying noise.

### 2.5. Pros & Cons

-   **Pros:**
    -   An elegant, architectural solution that directly targets the context corruption problem.
    -   Has the potential for the largest performance gains.
    -   Creates a more interpretable model, as we can inspect the gate scores.
-   **Cons:**
    -   Significantly more complex to implement than Idea 1.
    -   Adds new hyperparameters (the `gate_loss_weight`) that need tuning.
    -   Increases model complexity and computation time.

---

## Idea 3: The Algorithmic Change - "Annealed Decoding with Token Stickiness"

### 3.1. Core Concept

This idea addresses the "Unstable Decoding" problem by changing the *sampling algorithm*, not the model itself. We introduce a "stickiness" factor that makes it progressively harder for the model to change tokens that have remained stable for several steps. This encourages the model to commit to good predictions and prevents the chaotic "flickering" we observed.

### 3.2. Problem Addressed

-   **Primary:** Hypothesis 3 (Unstable Decoding Dynamics).

### 3.3. Detailed Implementation Plan

This change would occur entirely within the generation/analysis loop, for example, in a new `generate_sticky` function.

1.  **Modify the sampling loop in a new script (e.g., `analyze_sticky.py`):**
    -   Initialize a `stickiness_tensor` of shape `[batch_size, seq_len]` to all zeros.
    -   Initialize a `previous_tokens` tensor.
    -   Inside the reverse diffusion loop (from `t=999` down to `0`):
        -   Get the model's `logits` for the current `current_tokens`.
        -   **Apply the stickiness bonus:** Before the `softmax`, modify the logits to favor the existing tokens.
            ```python
            # A stickiness_factor hyperparameter would control the strength
            stickiness_bonus = torch.zeros_like(logits)
            # Add bonus to the logits of the tokens that are already there
            stickiness_bonus.scatter_(-1, previous_tokens.unsqueeze(-1), self.stickiness_tensor.unsqueeze(-1) * stickiness_factor)
            
            sticky_logits = logits + stickiness_bonus
            probabilities = torch.softmax(sticky_logits, dim=-1)
            ```
        -   Sample the `new_tokens` from these modified probabilities.
        -   **Update the stickiness tensor:**
            ```python
            # Compare new tokens to previous ones
            change_mask = (new_tokens != previous_tokens)
            
            # Increment stickiness for unchanged tokens
            self.stickiness_tensor[~change_mask] += 1
            
            # Reset stickiness for changed tokens
            self.stickiness_tensor[change_mask] = 0
            
            # Update previous_tokens for the next iteration
            previous_tokens = new_tokens
            ```

### 3.4. Expected Outcome & Metrics

-   **Prediction:** This will make the generation process far more stable and monotonic. The generated text quality should improve as the model builds upon a coherent foundation rather than constantly rewriting it.
-   **Success Metrics:**
    -   **Primary:** A qualitative analysis of the decoding trajectory (using the visualization script from Hyp. 3) should show a dramatic reduction in "flickering."
    -   **Secondary:** An improvement in sample quality, measured by human evaluation or more advanced metrics like MAUVE score.
    -   **Tertiary:** A potential improvement in diversity metrics (`Self-BLEU`, `Distinct-N`) as the model is less likely to get stuck in repetitive loops.

### 3.5. Pros & Cons

-   **Pros:**
    -   Requires no changes to the model architecture or training process. It's a purely algorithmic improvement at inference time.
    -   Can be tested on the existing, already-trained `final_uniform` model.
    -   Directly targets the observed instability.
-   **Cons:**
    -   Introduces new inference-time hyperparameters (`stickiness_factor`) that need tuning.
    -   It's a heuristic that fixes a symptom (instability) rather than the root cause in the model itself. It might prevent the model from correcting genuine early-stage mistakes.
