---
title: Is It Reasoning or Just a Fixed Bias?
author: Sriram Kiron
description: A mechanistic approach to reasoning.
image:
  url: https://docs.astro.build/assets/rays.webp
  alt: The Astro logo on a dark background with rainbow rays.
pubDate: 2026-01-12
tags: ["astro", "learning in public", "setbacks", "community"]
---
## tl;dr
I’m trying to answer whether LLMs actually perform multi-hop reasoning on ontology tasks, or whether any success they have shows something shallower.

The key discovery I made is that **models have a fixed "generalization" tendency *p*** that's task insensitive, meaning that they don't adapt to logical structure. Their accuracies in 1-hop (H1) and 2-hop (H2) reasoning roughly add up to 100%. This shows a bias innate to each model, as the *p* value varies model to model.

*See Table 2: Cross-model comparison*

My first piece of evidence to back this up is that **models perform conjunction detection, not compositional reasoning**.

- Set 1 (pure 2-hop reasoning): 3.5% accuracy
- Set 2 (4x salience, same logic): 6%
	- Salience (amount of parent mentions) alone doesn't help
- Set 4 (bridge entity added): 47.5%
	- Adding the shortcut works
- Set 5 (extra bridges): 43.5%
	- Similar to Set 4

My conclusion from this is that models detect conjunctions and don’t traverse ontologies. Basically, salience doesn’t really matter, but adding one bridge entity (conjunction) does.

Another piece of evidence I found is that analyzing overlap between H1 failures and H2 successes confirms task-insensitivity. Specifically for DeepSeek V3, which has a *p* of roughly 0.40, **73% of its H2 successes are also H1 failures** (they have the same output for H1 and H2).

If the model learns H2 (the parent) by properly reasoning through the ontology, it has to know H1 (the child). This 73% overlap between H1 failures and H2 successes shows a lack of genuine reasoning. It suggests that the model is skipping the chain and generalizing to a fixed level that aligns with the parent concept. And in the 27% of successes for H1 and H2, more than half show the model hedging by outputting both answers for each.

*See Table 4: Overlap Analysis of DeepSeek V3 H2 Successes*

My third piece of evidence comes from linear probing: the output-level probe shows 94.1% accuracy at layer 8 (an early layer), and it stays flat through layer 40. Probing is predicting *something* related to what the output is.

*See Table 5: Probing Results (Output Predictions by Layer)*

However, activation patching shows that we can't say exactly *what* probing is predicting. Patching early layer activations barely changes the final decision, making it more likely that the model's 2-hop accuracy (its *p* constant) is **distributed among many layers**. Probing isn't saying that models make a final decision early, just that it's detecting some abstract feature that correlates with the output. This still could be important, but I don't know what that feature is.

Another negative result I got was from SAE analysis. The top differential features I found encode surface patterns, like punctuation and formatting. I also found one high-effect feature which was about reasoning. But it’s very possible that all these features are just a result of shallow pattern matching. This would be consistent with my hypothesis that models have a fixed generalization tencency, and it would also add to my suspicion that *p* is distributed.

In short, apparent multi-hop success looks to be a result of fixed generalization bias, and it’s not a result of compositional reasoning or a decision made in any one component. I'm not close to done researching this, and I plan to look more into logit lens and other related works.

My code: https://github.com/Phineas1500/beyond-deduction
## Methods
### Dataset Design
I adapted the InAbHyD dataset for my experiment. I kept the original natural language data generation method, but I also added support for variations in first-order logic format with natural language entities, and in FOL format with symbolic (variable) entities.

I deviated more from the InAbHyD paper by creating specific sets of data using a matched pair design. For each logical structure, my code generates a H1 (1-hop) and H2 (2-hop) variant derived from the same ontology. This lets me accurately compare simple retrieval in H1 vs. potential reasoning in H2 (where models have to navigate a more complex structure). The characteristics of these datasets are shown in Table 1.
#### Table 1: **Summary of Matched Pair Datasets**

| Dataset   | Description         | Logical Structure       | Salience                 | Mechanism Tested                                                   |
| :-------- | :------------------ | :---------------------- | :----------------------- | :----------------------------------------------------------------- |
| **Set 1** | Pure Multi-hop      | Chain (2-hop)           | Low (1x parent mentions) | Genuine transitive reasoning (baseline)                            |
| **Set 2** | Salience Control    | Chain (2-hop)           | High (4x)                | Effect of token frequency on retrieval (without simplifying logic) |
| **Set 4** | INABHYD Replication | Convergent (Multi-path) | High (4x)                | Mimics original InAbHyD                                            |
| **Set 5** | Evidential Path     | Direct Membership       | High (4x)                | Conjunction detection (shortest path retrieval)                    |
>Caption: **Logical Structure** denotes the topology of the ontology graph; **Salience** indicates the presence of repetitive concept mentions or shortcuts designed to test behavior.

Feel free to ignore the fact that there’s no Set 3. It’s a result of my poor planning…
### Model Selection
After testing multiple models, I decided to use **Gemma 2 9B IT** for my experiments. The model is powerful enough to answer moderately advanced questions. Additionally, it's been the focus of extensive mechanistic interpretability work, with Google DeepMind training sparse auto-encoders on all the pre-trained model's layers (you can generally use PT-trained SAEs for IT models). This makes the model more accessible for MI. I ran inference for Gemma 2 9B IT (to find its accuracy on the dataset) on Modal with an H100.

I also conducted cross-model evaluation for accuracy on the dataset with Gemma 3 27B, DeepSeek V3, GPT-4o, and Llama 3 70B. I used Modal for Gemma 3 27B (H100), and I used OpenRouter for the other models.
### Evaluation
I ultimately evaluated the models on Set 1, **the pure multi-hop dataset**, so I could have a rigorous baseline for genuine transitive reasoning. The original InAbHyD dataset contains "shortcuts" like high token frequency (salience). I designed Set 1 with these conditions in mind:

- Only one logical path from child to parent, so no converging evidence
- Low salience (parent concept only mentioned once), preventing the model from making a prediction simply based on frequency
- No direct links between the child entity and the parent property

I compared strong accuracy (exact match) vs. weak accuracy (explains observations), as well as overlap between H1 failures and H2 successes. I tested 200 examples per condition to lower the statistical variance.

I also decided to use the natural language version of our dataset, because Gemma 2 9B is powerful enough to handle large amounts of text. Additionally, I felt that natural language was a more realistic way to evaluate reasoning than first-order logic.
### Linear Probing
I evaluated Gemma 2 9B with three types of probes: an output-level prediction probe, where I’m checking if the model outputs the child or parent; a concept-level probe, so I can tell if the model distinguishes between child and parent concepts; and a subsumption probe, which may enable me to see whether the model encodes an ontology structure.

I tested layers 8, 15, 20, 25, 30, 35, and 40, using bfloat16 precision with LayerNorm upcast to float32. I also used eager attention, because Flash Attention is incompatible with soft-capping for Gemma 2 9B. For this task, I ran activation extraction locally on an RTX 4090 GPU.
### Activation Patching
I also performed activation patching on Gemma 2 9B in order to test whether information in the early layers is causally important, not just correlational. I identified "success" (model correctly output the child concept) and "failure" (model incorrectly output the parent concept) examples. Then, I ran forward passes on both examples, replacing the activation of a "failure" example at a specific layer with a "success" example's activation at the same layer. I did patching for both the residual stream and the attention output components.
### SAEs
As mentioned in **Model Selection**, Gemma 2 9B has a vast array of pre-trained SAEs across all the models’ layers, collectively called Gemma Scope. I specifically used Gemma 2 9B's 16k residual stream SAE for layer 8 (not available for IT) and Gemma 2 9B IT's 16k residual stream SAE for layer 20.

I performed differential feature analysis, seeing what features distinguish parent-output vs. child-output examples. I then interpreted the features with the highest Cohen's d values on Neuronpedia.
## Results
### The Conservation Law
I evaluated five models on matched H1 and H2 pairs. The results are shown in Table 2.
#### Table 2: **Cross-model comparison**

| Model | H1 (child correct) | H2 (parent correct) | Sum | Implied p |
| :--- | :--- | :--- | :--- | :--- |
| **Gemma 2 9B** | 92% | 2.5% | 94.5% | 0.05 |
| **Gemma 3 27B** | 85% | 4.5% | 89.5% | 0.10 |
| **GPT-4o** | 84% | 14% | 98% | 0.15 |
| **DeepSeek V3** | 61% | 39% | 100% | 0.40 |
| **Llama 3 70B** | 42% | 42% | 84% | 0.50 |

>**Caption:** H1 accuracy + H2 accuracy ≈ 100% across almost all models.

This suggests that the models have a characteristic tendency *p* to output parent-level hypotheses, with H1 accuracy ≈ *1 - p* and H2 accuracy ≈ *p*. This is sort of a “Conservation Law.”
### Factorial Experiment
I also looked into Gemma 2 9B's performance on each of the datasets (in natural language), as you can see in Table 3.
#### Table 3: **Factorial Experiment Results (Gemma 2 9B)**

| Set            | Accuracy | Key Observation           |
| :------------- | :------- | :------------------------ |
| **Set 1 (H1)** | 91%      | Baseline retrieval        |
| **Set 1 (H2)** | 3.5%     | True multi-hop failure    |
| **Set 2 (H2)** | 6%       | Salience alone ≈ baseline |
| **Set 4 (H2)** | 47.5%    | INABHYD-style success     |
| **Set 5 (H2)** | 43.5%    | Evidential path works     |

>**Caption:** Set 5 >> Set 2 confirms conjunction detection, not compositional reasoning.

There wasn’t much of a jump in accuracy between Set 1 and Set 2 (which has an increased salience), but there was a large jump between Set 2 and Set 4 (which has identical salience but a bridge entity). Set 5 has even more bridge entities but identical salience, and it performed roughly the same as Set 4.

This suggests that increased saliency (parent mentions) doesn’t do much to increase accuracy, but adding a single bridge entity (one conjunction) does.
### Overlap
My “Conservation Law” predicts that models with higher *p* values aren't reasoning better but are instead simply calibrated to output parent-level hypotheses more frequently. To test this, I examined whether H2 successes represent genuinely different computational behavior from H1 failures, or whether they reflect identical outputs which are right in one place and wrong in another.

For DeepSeek V3 (*p* ≈ 0.40), I computed the overlap between H1 failures (where the model incorrectly outputted a parent-level hypothesis when child-level was correct) and H2 successes (where the model correctly outputted a parent-level hypothesis). If the model knew how to reason and generalize correctly, these sets should be largely disjoint and unrelated because the model would have to know the child first in order to reason the parent. That wasn't the case, though.
#### Table 4: **Overlap Analysis of DeepSeek V3 H2 Successes**

| Category                    | Count  | %        | Description                                                        |
| :-------------------------- | :----- | :------- | :----------------------------------------------------------------- |
| H2 successes ∩ H1 failures  | 57     | 73.1%    | Model outputs parent on both (same behavior, different task)       |
| H2 successes ∩ H1 successes | 21     | 26.9%    | Model outputs child on H1, parent on H2 (genuine task sensitivity) |
| **Total H2 successes**      | **78** | **100%** |                                                                    |

>**Caption:** 73% of reasoning successes are just the model outputting the parent concept regardless of the task logic.

As you can see in Table 4, 73.1% of DeepSeek V3's H2 successes were also H1 failures (57 data points). In most of DeepSeek V3's H2 "successes," the model produced the same parent-level output for H1. It didn't distinguish between H1 and H2, it just generalized no matter what and happened to be correct for H2.

And in the 26.9% of cases where H2 and H1 were successes, 0 of the 21 examples showed any genuine transitive reasoning chains. Over half even hedged by outputting both levels.

Results for Gemma 2 9B are less informative because of its lack of H2 successes, but it still showed a lack of task sensitivity.
### Probing
I started my main MI work with linear probing. The results for three important layers are shown in Table 5.

#### Table 5: **Probing Results (Output Predictions by Layer)**

| Layer | Balanced Accuracy | Initial Interpretation   |
| :---- | :---------------- | :----------------------- |
| 8     | 94.1% ± 6.9%      | Decision locked in early |
| 20    | 93.2% ± 5.6%      | No computation change    |
| 40    | 93.9% ± 6.3%      | Final output determined  |

>**Caption:** Barely any change in accuracy, seems to be decided early.

The flat accuracy trajectory from layers 8 to 40 *suggests* that the generalization decision is representational (fixed by layer 8) rather than the result of iterative reasoning in deeper layers. The story gets more complicated, though.
### Activation Patching
To test whether Layer 8 information is causally important, I performed activation patching on Gemma 2 9B.
#### Table 6: **Patching Results**

| Component        | Layer | Patching Effect |
| ---------------- | ----- | --------------- |
| Residual Stream  | 8     | 0%              |
| Residual Stream  | 12    | 0%              |
| Attention Output | 12    | 25%             |

>**Caption**: Patching has barely any effect.

Despite 94.1% probe accuracy at Layer 8, patching the residual stream has 0% effect on model output. The probe detects information that **exists but isn't used**. The probe reads information that is correlated to the decision, but not the decision itself. This heavily suggests that the *p* constant is distributed among many layers, instead of being decided early on like I first suspected after probing.
### SAE Features
For our work with SAEs, I prioritized layers 8 and 20. I split the data I used into two sets: "gold" and control. Control data is H1 success and H2 failure, while "gold” data is where Gemma outputs parent-level hypotheses, representing its rare generalization behavior (H1 failure, H2 success). I discussed this type of data in the **Overlap** section, as it's common for DeepSeek V3 to generate it. It's rare for Gemma 2 9B to output it, though, and I had to generate 1,500 data points to get 53 gold examples.

I extracted the top 20 differentially-activating features at each layer and retrieved their interpretations from Neuronpedia. Some representative findings are shown on Table 7.
#### Table 6: **Neuronpedia Interpretations of Top SAE Features**

| Layer | Feature | Direction | Neuronpedia Interpretation |
| :--- | :--- | :--- | :--- |
| 8 | 2639 | Gold | "problem-solving/reasoning task framing" |
| 8 | 5314 | Control | "punctuation marks and their frequency" |
| 8 | 9112 | Control | "punctuation marks and sentence boundaries" |
| 20 | 1831 | Gold | "molecular biology and genetics terminology" |
| 20 | 6456 | Control | "programming-related syntax & mathematical expressions" |
| 20 | 3408 | Control | "personal & familial identification or relationships" |

>**Caption:** Features detecting "reasoning" (gold) or "syntax" (control) suggest the model is classifying the *format* of the prompt, not its logical structure.

The SAE results are less clear than what I’d been seeing so far, but they might suggest that the model looks more at how the prompt is formatted rather than the logical structure of it.
## Analysis
### What the Conservation Law Reveals
Through my cross-model analysis, I found a mathematical relationship between H1 and H2 accuracies, where the two values sum up to roughly 100%. This shows that models have a fixed generalization parameter *p* that’s task-insensitive. The model doesn’t modify its reasoning based on a prompt’s logical structure, it just has a fixed bias.

This is supported by my factorial experiment, which confirmed conjunction detection over compositional reasoning. *p* reflects a probability that the model can find a pattern, not how well it can reason.
### Probing: A Bit Misleading
Probing initially seemed to show that the model’s output-level decision is cemented by layer 8 (19% depth) at 94.1% accuracy. This isn't the case, though, as activation patching shows that this information picked up by probing has **no causal effect**. The probe simply finds a correlation, not the steering wheel itself.
### A Lesson From SAE Analysis
Even though SAE results weren’t positive, I still think they could be informative. SAE features that distinguish gold from control data seem to encode surface-level patterns, not full decisions. This is especially seen by all the high effect features for punctuation.

And feature 2639, which is for reasoning task framing, might not be contradictory, since it quite possibly just detects task format, not the reasoning process. 

This finding is consistent with everything else I’ve observed. If the decision is made through weak pattern matching, you wouldn’t expect to find clean generalization level SAE features. These results further support our argument after activation patching that the overarching decision is distributed across many weak features.
### Synthesis
My distinct areas of analysis converged and point to the same conclusion: models use very shallow pattern matching with a fixed generalization bias, not compositional reasoning. I will fully note that I made this conclusion after conducting tests on my specific non-deductive reasoning data, but I still think I can take these findings further.

The *p* parameter is determined at encoding (early layer 8), it’s task-insensitive (as seen in the overlap analysis), and it’s not cleanly causal at early layers or captured by SAE features (meaning it's probably distributed). I'm going to continue my research because I think conclusively proving the fixed bias in a model’s reasoning will do a lot to show LLMs’ limitations and potentially where we can improve them for the better.
## Works Cited
Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, and Neel Nanda. “Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2”. In: arXiv preprint arXiv:2408.05147 (2024).

nalf3in2 and Hugging Face Community. Flash attention 2 is not working. https://huggingface.co/google/gemma-2-9b-it/discussions/9. Discussion on the Hugging Face repository for google/gemma-2-9b-it. 2024. url: https://huggingface.co/google/gemma-2-9b-it/discussions/9.

Neuronpedia Community and Google Deepmind. GEMMA-2-9B. https://www.neuronpedia.org/gemma-2-9b. Interactive model interpretability page for the GEMMA-2-9B model on Neuronpedia. 2025. url: https://www.neuronpedia.org/gemma-2-9b.

Yunxin Sun and Abulhair Saparov. “Language Models Do Not Follow Occam’s Razor: A
Benchmark for Inductive and Abductive Reasoning”. In: arXiv preprint arXiv:2509.03345
(2025).