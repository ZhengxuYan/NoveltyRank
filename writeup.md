## Tasks And Methods
- **Classification task**: predict the novelty label (`label` column) from a single arXiv abstract. SFT uses the cleaned train/test splits under `data_cache/whole_dataset/train_sft_data` and `data_cache/whole_dataset/test_sft_data`, while the classification-DPO pipeline converts each labeled example into preferred/rejected completions via `create_classification_example`.
- **Comparison task**: given a positive (novel) and negative (non-novel) abstract, select the more novel item. Pairs come from `build_comparison_pairs`, which cycles positives across all negatives to upsample minority examples.
- **Comparison task**: given a positive (novel) and negative (non-novel) abstract, select the more novel item. Pairs come from `build_comparison_pairs`, which cycles positives across all negatives to upsample minority examples.
- Note: inputs used in the examples and error analysis are the `Title` and `Abstract`. `Labels:` are shown as `predicted/true`. `Similarity Score` reports `max` and `avg` similarity.
- **Method stack**:
  - *Supervised fine-tuning (SFT)* fine-tunes Qwen-4B with cross-entropy and evaluates through `AccuracyOnLabeledTestSetEvaluator` (see `scripts/Qwen_4B/train/sft.py`).
  - *Supervised fine-tuning (SFT)* fine-tunes Qwen-4B with cross-entropy and evaluates through `AccuracyOnLabeledTestSetEvaluator` (see `scripts/Qwen_4B/train/sft.py`).
  - *Classification-DPO* applies `generate_classification_dpo_pairs` to build balanced preference datasets before DPO training (`scripts/Qwen_4B/train/dpo.py`).
	- *Comparison-DPO* optimizes on pairwise preferences produced by `build_comparison_pairs`; rewards are derived from a frozen reference model.

## Whole-Dataset Experiments (WHOLE_DATASET)
- **SFT**
	- Result dir: `results/noveltyrank_sft_qwen4b`
	- Final sampler: `tinker://677ae6cd-3bae-562c-86a8-7d6f2860ec34:train:0/sampler_weights/final`
	- Test metrics: accuracy 0.898, precision 0.273, recall 0.065, F1 0.105 (temperature 0.0, limit 500)
- **Classification-DPO**
	- Result dir: `results/noveltyrank_dpo_qwen4b_classification`
	- Final sampler: `tinker://19683cd1-f5fb-5c27-a4ab-db46ecb07475:train:0/sampler_weights/final`
	- Test metrics: accuracy 0.864, precision 0.194, recall 0.152, F1 0.171 (temperature 0.0, limit 500)
- **Comparison-DPO**
	- Result dir: `results/noveltyrank_dpo_qwen4b_comparison`
	- Final sampler: `tinker://2e30d03e-ef52-528e-b88d-eb75c95c856c:train:0/sampler_weights/final`
	- Test metrics: accuracy 0.586 (temperature 0.0, limit 1000)
- **GPT-5 Frontier API eval**
  - Setup: zero-shot inference via frontier GPT-5 classification endpoint on the full test split
  - Test metrics: accuracy 0.7036, precision 0.2449, recall 0.6606, F1 0.3573

With only ~12% positives in this test set, accuracy alone can overstate performance: a majority-class guesser already reaches 0.88 accuracy. The reported precision/recall gaps highlight how SFT prioritizes the dominant negative class, while DPO variants improve balance modestly but still miss many positives. Even the comparison DPO run, which climbed from ~0.395 starting accuracy to 0.586, only edges out random choice; recall-oriented tuning and stronger positive-signal augmentation remain necessary. 

Because the three approaches saturate at low recall, we attriute it to two reason: 1. hard task  2. small model.
Our next steps narrow the scope to category-specific slices like CS_CV and scale up to larger backbones to probe whether richer capacity exposes more positive signal.

## Base-Model Benchmarks (No Fine-Tuning)
- **Qwen 4B @ T=0**: classification accuracy 0.597 (F1 0.392) and comparison accuracy 0.547 (F1 0.456).
- **Qwen 235B @ T=0**: classification accuracy 0.487 (F1 0.403) and comparison accuracy 0.570 (F1 0.606).

Sweeping temperatures between 0.0 and 1.0 in both files nudged accuracies by at most ~0.02 and left precision/recall almost unchanged, indicating these base models are largely temperature-insensitive for this evaluation.

Despite the 235B model’s size advantage, its classification accuracy lags the 4B variant and only ekes out modest gains on comparison; scale alone fails to unlock high recall, reinforcing that the underlying task remains intrinsically hard under current prompts and data.

## Dataset Distribution Investigation
- **Train split overview:** 32,003 examples with 4,547 positives (14.2%); detailed counts in `data_cache/analysis/outputs/train_primary_category_summary.csv` highlight:
  - `cs.CV`: 7,856 total / 1,619 positive (20.6% rate)
  - `cs.CL`: 6,538 / 1,455 (22.3%)
  - `cs.LG`: 7,032 / 1,022 (14.5%)
  - `cs.RO`: 4,759 / 109 (2.3%)
  - `cs.CR`: 3,450 / 120 (3.5%)
  - `cs.AI`: 2,368 / 222 (9.4%)
- **Test split overview:** 10,889 examples with 1,358 positives (12.5%); `data_cache/analysis/outputs/test_primary_category_summary.csv` shows the same leaders but with leaner positive ratios:
  - `cs.CV`: 3,358 / 631 (18.8%)
  - `cs.CL`: 1,755 / 326 (18.6%)
  - `cs.LG`: 2,206 / 243 (11.0%)
  - `cs.RO`: 1,385 / 45 (3.2%)
  - `cs.CR`: 1,401 / 48 (3.4%)
  - `cs.AI`: 784 / 65 (8.3%)

    Computer vision supplies the largest sample pool and keeps the most stable positive ratio across splits, making it the best-balanced slice for targeted experimentation.

## Category Focus: CS_CV
- **SFT**
	- Result dir: `results/noveltyrank_sft_qwen4b_cv`
	- Fnial Sampler: `tinker://2e566352-bd3a-5c10-8e5a-2743d49bc353:train:0/sampler_weights/final`
	- Test metrics (limit 500, T=0.0): accuracy 0.758, precision 0.255, recall 0.149, F1 0.188
- **Classification-DPO**
	- Result dir: `results/noveltyrank_dpo_qwen4b_classification_cs_cv`
	- Final sampler: `tinker://e88323f5-71b3-5a81-a7e8-29e34c7ff873:train:0/sampler_weights/final`
  - Test metrics (limit 500, T=0.0): accuracy 0.704, precision 0.254, recall 0.298, F1 0.275
- **Comparison-DPO**
	- Result dir: `results/dpo_comparison_cv`
  - Final sampler: `tinker://f3ae720f-f1df-5ce8-92e5-300dd59b1b5f:train:0/sampler_weights/final`
  - Test metrics (limit 500, T=0.0): accuracy 0.598

Classification-DPO increases recall relative to SFT while keeping precision similar; Comparison-DPO slightly improves pairwise ranking. Further analysis should persist raw predictions for error inspection.

## Conduct Error Analysis ##

Examples originate from running `scripts/Qwen_4B/test/analyze_classification_errors.py` on the CS_CV SFT sampler (`tinker://2e566352-bd3a-5c10-8e5a-2743d49bc353:train:0/sampler_weights/final`, the SFT run described in the section above; limit 500, T=0.0).

False Positives (first 3):
[1] Title: Referring Expression Instance Retrieval and A Strong End-to-End Baseline
    Abstract:
      Using natural language to query visual information is a fundamental need in real-world applications. Text-Image Retrieval (TIR) retrieves a target image from a gallery based on an image-level description, while Referring Expression Comprehension (REC) localizes a target object within a given image using an instance-level description. However, real-world applications often present more complex demands. Users typically query an instance-level description across a large gallery and expect to receive both relevant image and the corresponding instance location. In such scenarios, TIR struggles with fine-grained descriptions and object-level localization, while REC is limited in its ability to efficiently search large galleries and lacks an effective ranking mechanism. In this paper, we introduce a new task called \textbf{Referring Expression Instance Retrieval (REIR)}, which supports both instance-level retrieval and localization based on fine-grained referring expressions. First, we propose a large-scale benchmark for REIR, named REIRCOCO, constructed by prompting advanced vision-language models to generate high-quality referring expressions for instances in the MSCOCO and RefCOCO datasets. Second, we present a baseline method, Contrastive Language-Instance Alignment with Relation Experts (CLARE), which employs a dual-stream architecture to address REIR in an end-to-end manner. Given a referring expression, the textual branch encodes it into a query embedding. The visual branch detects candidate objects and extracts their instance-level visual features. The most similar candidate to the query is selected for bounding box prediction. CLARE is first trained on object detection and REC datasets to establish initial grounding capabilities, then optimized via Contrastive Language-Instance Alignment (CLIA) for improved retrieval across images. We will release our code and benchmark publicly.
    Labels: predicted=1 / true=0
    Similarity Score: max=0.680411820946626 avg=0.5514792378107412
[2] Title: Towards Explicit Geometry-Reflectance Collaboration for Generalized   LiDAR Segmentation in Adverse Weather
    Abstract:
      Existing LiDAR semantic segmentation models often suffer from decreased accuracy when exposed to adverse weather conditions. Recent methods addressing this issue focus on enhancing training data through weather simulation or universal augmentation techniques. However, few works have studied the negative impacts caused by the heterogeneous domain shifts in the geometric structure and reflectance intensity of point clouds. In this paper, we delve into this challenge and address it with a novel Geometry-Reflectance Collaboration (GRC) framework that explicitly separates feature extraction for geometry and reflectance. Specifically, GRC employs a dual-branch architecture designed to independently process geometric and reflectance features initially, thereby capitalizing on their distinct characteristic. Then, GRC adopts a robust multi-level feature collaboration module to suppress redundant and unreliable information from both branches. Consequently, without complex simulation or augmentation, our method effectively extracts intrinsic information about the scene while suppressing interference, thus achieving better robustness and generalization in adverse weather conditions. We demonstrate the effectiveness of GRC through comprehensive experiments on challenging benchmarks, showing that our method outperforms previous approaches and establishes new state-of-the-art results.
    Labels: predicted=1 / true=0
    Similarity Score: max=0.5686580604210486 avg=0.5148353317664388
[3] Title: SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and   Hearing
    Abstract:
      3D spatial reasoning in dynamic, audio-visual environments is a cornerstone of human cognition yet remains largely unexplored by existing Audio-Visual Large Language Models (AV-LLMs) and benchmarks, which predominantly focus on static or 2D scenes. We introduce SAVVY-Bench, the first benchmark for 3D spatial reasoning in dynamic scenes with synchronized spatial audio. SAVVY-Bench is comprised of thousands of relationships involving static and moving objects, and requires fine-grained temporal grounding, consistent 3D localization, and multi-modal annotation. To tackle this challenge, we propose SAVVY, a novel training-free reasoning pipeline that consists of two stages: (i) Egocentric Spatial Tracks Estimation, which leverages AV-LLMs as well as other audio-visual methods to track the trajectories of key objects related to the query using both visual and spatial audio cues, and (ii) Dynamic Global Map Construction, which aggregates multi-modal queried object trajectories and converts them into a unified global dynamic map. Using the constructed map, a final QA answer is obtained through a coordinate transformation that aligns the global map with the queried viewpoint. Empirical evaluation demonstrates that SAVVY substantially enhances performance of state-of-the-art AV-LLMs, setting a new standard and stage for approaching dynamic 3D spatial reasoning in AV-LLMs.
    Labels: predicted=1 / true=0
    Similarity Score: max=0.6269728847303679 avg=0.5745204375951349


False Negatives (first 3):
[1] Title: Deterministic Object Pose Confidence Region Estimation
    Abstract:
      6D pose confidence region estimation has emerged as a critical direction, aiming to perform uncertainty quantification for assessing the reliability of estimated poses. However, current sampling-based approach suffers from critical limitations that severely impede their practical deployment: 1) the sampling speed significantly decreases as the number of samples increases. 2) the derived confidence regions are often excessively large. To address these challenges, we propose a deterministic and efficient method for estimating pose confidence regions. Our approach uses inductive conformal prediction to calibrate the deterministically regressed Gaussian keypoint distributions into 2D keypoint confidence regions. We then leverage the implicit function theorem to propagate these keypoint confidence regions directly into 6D pose confidence regions. This method avoids the inefficiency and inflated region sizes associated with sampling and ensembling. It provides compact confidence regions that cover the ground-truth poses with a user-defined confidence level. Experimental results on the LineMOD Occlusion and SPEED datasets show that our method achieves higher pose estimation accuracy with reduced computational time. For the same coverage rate, our method yields significantly smaller confidence region volumes, reducing them by up to 99.9\% for rotations and 99.8\% for translations. The code will be available soon.
    Labels: predicted=0 / true=1
    Similarity Score: max=0.4927147953280572 avg=0.44649690736146536
[2] Title: UniCon: Unidirectional Information Flow for Effective Control of   Large-Scale Diffusion Models
    Abstract:
      We introduce UniCon, a novel architecture designed to enhance control and efficiency in training adapters for large-scale diffusion models. Unlike existing methods that rely on bidirectional interaction between the diffusion model and control adapter, UniCon implements a unidirectional flow from the diffusion network to the adapter, allowing the adapter alone to generate the final output. UniCon reduces computational demands by eliminating the need for the diffusion model to compute and store gradients during adapter training. Our results indicate that UniCon reduces GPU memory usage by one-third and increases training speed by 2.3 times, while maintaining the same adapter parameter size. Additionally, without requiring extra computational resources, UniCon enables the training of adapters with double the parameter volume of existing ControlNets. In a series of image conditional generation tasks, UniCon has demonstrated precise responsiveness to control inputs and exceptional generation capabilities.
    Labels: predicted=0 / true=1
    Similarity Score: max=0.6267436656682516 avg=0.5814269075971615
[3] Title: GGTalker: Talking Head Systhesis with Generalizable Gaussian Priors and   Identity-Specific Adaptation
    Abstract:
      Creating high-quality, generalizable speech-driven 3D talking heads remains a persistent challenge. Previous methods achieve satisfactory results for fixed viewpoints and small-scale audio variations, but they struggle with large head rotations and out-of-distribution (OOD) audio. Moreover, they are constrained by the need for time-consuming, identity-specific training. We believe the core issue lies in the lack of sufficient 3D priors, which limits the extrapolation capabilities of synthesized talking heads. To address this, we propose GGTalker, which synthesizes talking heads through a combination of generalizable priors and identity-specific adaptation. We introduce a two-stage Prior-Adaptation training strategy to learn Gaussian head priors and adapt to individual characteristics. We train Audio-Expression and Expression-Visual priors to capture the universal patterns of lip movements and the general distribution of head textures. During the Customized Adaptation, individual speaking styles and texture details are precisely modeled. Additionally, we introduce a color MLP to generate fine-grained, motion-aligned textures and a Body Inpainter to blend rendered results with the background, producing indistinguishable, photorealistic video frames. Comprehensive experiments show that GGTalker achieves state-of-the-art performance in rendering quality, 3D consistency, lip-sync accuracy, and training efficiency.
    Labels: predicted=0 / true=1
    Similarity Score: max=0.6805159661291962 avg=0.6093042309503641

### Weaknesses Observed From FP/FN Cases
- False positives cluster around trendy CV topics that sound innovative but deliver incremental gains; the model latches onto buzzwords or "new benchmark" framing instead of verifying substantive novelty.
- False negatives are technically substantial papers whose abstracts bury innovations in dense math or engineering language; when contributions arrive late or read as classical analysis, the model defaults to the majority class.
- Predictions over-weight the opening paragraphs and familiar phrases such as "dual-branch" or "benchmark"; once those cues appear the model over-penalizes, exposing a recall gap on method-heavy work.
- Similarity heuristics offer little separation between FP and FN cases, reinforcing that deeper semantic reasoning is required beyond max/avg similarity signals.

Overall, the key issue is that we lack sufficient feature to help the model better understand novelty. Similarity scores alone are insufficient, and we need to explore richer metadata and context to guide the model's judgments.

### taking Next Steps
1. remove comparison DPO for now, focus on classification SFT and classification DPO.(considering the limited improvement from comparison DPO and the difficulty of error analysis)
2. Use a detailed similarity report to help error analysis and data augmentation instead of only using similarity scores.


## Add New Feature: Similarity Report
We extract the top-K similar papers from arXiv for each target example in the dataset (stored in a dedicated column). A Qwen-235B model then generates a detailed similarity report informed by the target paper and its neighbours. Each report is instructed to cover four sections:

1. **Shared Themes**
  - Focus on topics, methods, tasks, data types, or motivations that appear explicitly in the texts.
2. **Overlap Snapshot**
  - Summarise key overlap areas in two to three sentences.
  - Mention which arXiv IDs drive each overlap signal without analysing papers individually.
3. **Distinctive Aspects**
  - Bullet list of target-specific ideas or claims absent from the similar papers.
  - Ground every bullet in explicit differences observed in the abstracts.
4. **Novelty Verdict**
  - Conclude with High/Medium/Low and a one-sentence justification tied to the aggregated evidence.


### Results (Focusing on CS_CV)

- **SFT**
  - Result dir: `results/new_model_sft_cv`
  - Final sampler: `tinker://4ba31574-b75b-52fc-a87a-408e984590d0:train:0/sampler_weights/final`
  - Test metrics (limit 500, T=0.0): accuracy 0.792, precision 0.414, recall 0.315, F1 0.358
- **Classification-DPO (lr=5e-4)**
  - Result dir: `results/new_model_dpo_cv`
  - Final sampler: `tinker://ad85b617-2ed7-5dde-b0f7-9302423902a0:train:0/sampler_weights/final`
  - Test metrics (limit 500, T=0.0): accuracy 0.568, precision 0.310, recall 0.338, F1 0.324, unresolved 124/500
- **SFT + DPO**
  - Result dir: `results/new_model_dpo_cv_sftinit`
  - Best sampler: `tinker://9a82def9-793e-51f8-8a7a-cd23781cbdd4:train:0/sampler_weights/000310`
  - Test metrics (limit 500, T=0.0): {'test_accuracy': 0.728, 'test_precision': 0.33064516129032256, 'test_recall': 0.43617021276595747, 'test_F1': 0.3761467889908257}

### Observations
- Similarity reports help SFT improve precision and recall significantly, indicating that richer context aids the model's understanding of novelty.
- Classification-DPO with similarity reports shows moderate accuracy but balanced precision/recall, suggesting that the additional context helps mitigate overfitting to the majority class.



### Latest Results Summary

- without similarity report:
- **SFT**
	- Result dir: `results/sft_cv`
	- Fnial Sampler: `tinker://b134fa47-0ac6-57bc-b8c7-9cf138a3ecaa:train:0/sampler_weights/final`
  - Test metrics (limit 500, T=0.0): accuracy 0.776, precision 0.350, recall 0.223, F1 0.273
- **Classification-DPO**
	- Result dir: `results/dpo_classification_cv`
	- Final sampler: `tinker://61ef747f-c41d-5587-a754-771e2b1e114e:train:0/sampler_weights/final`
  - Test metrics (limit 500, T=0.0): {'test_accuracy': 0.35, 'test_precision': 0.22033898305084745, 'test_recall': 0.9680851063829787, 'test_F1': 0.3589743589743589}
- **SFT + DPO**
  - Result dir: `results/dpo_classification_cv_sftinit`
  - Best sampler: `tinker://ff01d0b0-c39e-57f6-ba9d-552f229feb97:train:0/sampler_weights/final`
  - Test metrics (limit 500, T=0.0): {'test_accuracy': 0.744, 'test_precision': 0.31521739130434784, 'test_recall': 0.30851063829787234, 'test_F1': 0.3118279569892473}

- with similarity report:
- **SFT**
  - Result dir: `results/sft_cv_sim`
  - Final sampler: `tinker://a365c343-65f2-50b7-8276-4bade5f51896:train:0/sampler_weights/final`
  - Test metrics (limit 500, T=0.0): accuracy 0.750, precision 0.304, recall 0.255, F1 0.277
- **Classification-DPO**
  - Result dir: `results/dpo_classification_cv_sim`
  - Final sampler: `tinker://6903b11c-9d7d-52d6-9229-68e6b4ce0d56:train:0/sampler_weights/final`
  - Test metrics (limit 500, T=0.0): {'test_accuracy': 0.386, 'test_precision': 0.22900763358778625, 'test_recall': 0.9574468085106383, 'test_F1': 0.36960985626283366}
- **SFT + DPO**
  - Result dir: `results/dpo_classification_cv_sftinit_sim`
  - Best sampler: `tinker://d55a8693-af3c-5e69-9006-d62ab4a34aed:train:0/sampler_weights/final`
  - Test metrics (limit 500, T=0.0): 

python scripts/Qwen_4B/test/test_classification.py \
   --category CS_CV \
   --limit 500 \
   --model-path tinker://4ba31574-b75b-52fc-a87a-408e984590d0:train:0/sampler_weights/final

