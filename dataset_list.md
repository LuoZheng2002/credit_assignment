 # 0_summary.pdf

- Training Set
  - No external datasets used; document is an executive summary comparing methods. (No experiments reported that use specific public datasets.)

- Evaluation Set
  - No external datasets used; summary paper consolidates methods and references other papers which contain experiments.

# 1_igpo.pdf

 - Training Set
  - NQ (Natural Questions) — open-domain QA; used as in-domain training data; (Kwiatkowski et al., 2019). Referenced in Experiments (Sec. 4.1).
  - TQ (TriviaQA) — QA dataset; used as in-domain training data; (Joshi et al., 2017). Referenced in Experiments (Sec. 4.1).
  - HotpotQA — multi-hop QA dataset; used as in-domain training data; (Yang et al., 2018). Referenced in Experiments (Sec. 4.1).
  - 2Wiki — multi-hop QA / reasoning dataset (2WikiMultiHopQA); used as in-domain training data; (Ho et al., 2020). Referenced in Experiments (Sec. 4.1).

 - Evaluation Set
  - MusiQue — multi-hop composition QA (MusiQue, Trivedi et al., 2022); used as out-of-domain evaluation (Sec. 4.1).
  - Bamboogle — OOD benchmark (Press et al., 2022) used as out-of-domain evaluation (Sec. 4.1).
  - PopQA — PopQA dataset (Mallen et al., 2022); used as out-of-domain evaluation (Sec. 4.1).

Notes:
  - Paper uses a search tool (Google Search API) as the environment/tool for agent rollouts (Implementation details, Sec. 4.1).
  - Metrics: word-level F1 reported. The paper reports experiments on in-domain (NQ, TQ, HotpotQA, 2Wiki) and out-of-domain (MusiQue, Bamboogle, PopQA) benchmarks (Table 1).

# 2_scar.pdf

 - Training Set
  - IMDB (movie reviews) — sentiment classification dataset used for sentiment-control experiments (Maas et al., 2011). Used to fine-tune/init policy and for RLHF training (Sec. 4.1).
  - Reddit TL;DR (filtered version from Stiennon et al., 2020) — summarization dataset (approx. 116K train / 6K val / 6K test) used for text summarization experiments (Sec. 4.1).
  - Anthropic HH (helpfulness subset of Helpful and Harmless) — instruction-following preference data used for instruction-tuning experiments (43K prompts) (Bai et al., 2022). Used to train reward model and RLHF (Sec. 4.1).

 - Evaluation Set
  - IMDB test split — reported test reward for sentiment control (Table 1).
  - TL;DR test split — reported test results and LLM-as-judge evaluations on 1K sampled summaries (Sec. 4.1, Table 1).
  - Anthropic HH held-out set / AlpacaEval samples — used AlpacaEval/gpt-4-turbo to evaluate instruction-tuning outputs (Sec. 4.1, Table 2).

Notes:
  - SCAR experiments were run on three tasks (sentiment control, summarization, instruction tuning) with respective public datasets listed above (Sec. 4.1).
  - For summarization, the paper used Pythia-1B model and a learned 1B reward model trained on 92K human preference pairs (Sec. 4.1).

# 3_grpo_lambda.pdf

 - Training Set
  - GSM8K — grade school math dataset used for RL finetuning (Cobbe et al., 2021). Referenced as one of the RL finetuning datasets (Sec. 4).
  - Math-12K — public math dataset (lightman et al. extraction) used for RL finetuning (Sec. 4).
  - MathRL-16K — RL math dataset used for RL finetuning (HuggingFace dataset reference) (Sec. 4).
  - ORZ MATH-57K — larger math dataset used for RL finetuning (Hu et al., 2025) (Sec. 4).

 - Evaluation Set
  - AIME24 — evaluation benchmark for competition math (used in tables/benchmarks, Sec. 4.3).
  - AMC — AMC benchmark (evaluation; Sec. 4.3).
  - OlympiadBench / OlympiadMath — olympiad-level benchmark used for evaluation (He et al., 2024) (Sec. 4.3).
  - Math500 — benchmark (Hendrycks et al., 2021) used for evaluation (Sec. 4.3).
  - MinervaMath — Minerva-style benchmark (Lewkowycz et al., 2022) used for evaluation (Sec. 4.3).

 Notes:
  - Experiments cover multiple model sizes (1.5B, 3B, 7B) and architectures (Qwen2.5 and LLaMA-3.1 variants) and evaluate on several math benchmarks (Table 1, Sec. 4.3).
  - Training datasets and benchmarks are cited with HuggingFace dataset links in the appendix (see Sec. 4 and Appendix C/E).

# 4_capo.pdf

 - Training Set
  - NuminaMath-CoT — large CoT math dataset used for SFT pretraining (NuminaMath, 860K samples) referenced for Qwen experiments (Sec. 4.1).
  - MATH — mathematical reasoning dataset used for RL and evaluation (Hendrycks et al., 2021). Used for RL on Llama and Qwen variants (Sec. 4.1).
  - DAPO‑Math — dataset used for RL finetuning for some experiments (Yu et al., 2025) (Sec. 4.1).

 - Evaluation Set
  - MATH (held-out test) — in-distribution math benchmark (Hendrycks et al., 2021) (Sec. 4.1, Tables 2–3).
  - OlympiadBench — olympiad-level math benchmark (He et al., 2024) (Sec. 4.1, Tables 2–3).
  - AMC2023 — AMC benchmark from the Mathematical Association of America (2023) (Sec. 4.1).
  - AIME2024 — AIME benchmark (Mathematical Association of America, 2024) (Sec. 4.1).
  - GPQA-diamond, ARC-c, MMLU‑Pro — out-of-distribution/general reasoning benchmarks used for OOD evaluation (Sec. 4.1, Tables 2–3).

 Notes:
  - CAPO uses an LLM-as-GenPRM (e.g., Qwen2.5-72B-Instruct or Llama-3-70B-Instruct) to generate per-step critiques; voting over multiple critiques (N=4 or 8) is applied (Sec. 3.1, 4.1).
  - Experiments run on Llama-3-1B/3B and Qwen2.5-1.5B/7B backbones; results reported as Pass@1 (Tables 2–3).

# 5_int.pdf

 - Training Set
  - Polaris, AceReason-Math, Omni-MATH subsets — curated hard-problem pools filtered for zero accuracy under many rollouts; used to construct Dhard and training interventions (Sec. 5.1).
  - DeepScaleR subset — used for designing SFT configurations and ablations (Sec. 5.1, Table 1).

 - Evaluation Set
  - IMO-AnswerBench — IMO-level benchmark (Luong et al., 2025); used for evaluation (Pass@1) (Sec. 5.5, Table 2–3).
  - HMMT 2025 Nov — held-out competition benchmark used for evaluation (Sec. 5.5, Table 3).
  - AMO-Bench, Apex Shortlist — additional standardized math benchmarks used for evaluation (Sec. 5.5, Table 3).
  - Dtest_hard — i.i.d. held-out hard-problem test set sampled from Dhard (64 problems) (Sec. 5.1).

 Notes:
  - InT (Intervention Training) generates single-step interventions by self-verification against reference solutions and uses SFT on (prefix + intervention) without suffix, then RL (Algorithm 1, Sec. 4).
  - Models/roots: experiments use Qwen3-4B-Instruct and larger Qwen/Llama variants as generators/verifiers; intervention oracle experiments also use Gemini-2.5-Pro (Sec. 3–5).

# 6_spark.pdf

 - Training Set
  - Skywork-OR1-RL-Data — 8k/17k subsets used for PRM training and RL training respectively (He et al., 2025); used to generate synthetic verification examples via generator-verifier framework (Sec. 3.1, 4.2).
  - Eurus-2-SFT-Data — 113K SFT problems with structured solutions used to teach policy formatting before RL (Appendix D, Sec. 4.2).

 - Evaluation Set
  - ProcessBench — benchmark for identifying erroneous steps in mathematical reasoning used to evaluate PRMs (Sec. 3.1, Table 2).
  - MATH-500, AIME 2024/2025, AMC 2023, OlympiadBench, MinervaMath — mathematical reasoning benchmarks used for RL evaluation (Tables 1–3, Sec. 4.3).

 Notes:
  - SPARK produces synthetic step-level verification data using inference-time scaling (self-consistency, meta-critique, hybrid) and trains generative PRMs (ORM, PRM, PRM-CoT) from Qwen2.5-14B-Instruct (Sec. 2–3).
  - PRMs (trained with step-level consistency) are used as reward models in GRPO-based RL; authors analyze multiple reward formulations and reward-hacking modes (Sec. 4).

# 7_genprm.pdf

 - Training Set
  - MATH dataset (subset) — used to synthesize 23K training examples for GenPRM via MC estimation, RPE, and rationale synthesis (Sec. 3.2, Appendix A) — authors note GenPRM trained on 23K MATH samples (Sec. 4.2).
  - Skywork-OR1-RL-Data and other public math corpora referenced for solution generation and evaluation (Sec. 3.2, A.1).

 - Evaluation Set
  - ProcessBench — benchmark for detecting erroneous reasoning steps; primary evaluation for PRM performance (Tables 1–3, Sec. 4.2).
  - MATH, AMC23, AIME24, MinervaMath, AIME25 — used for BoN and policy-model test-time scaling experiments (Sec. 4.3, Table 2–3).

 Notes:
  - GenPRM is a generative PRM that produces CoT and code-verification rationales, uses Relative Progress Estimation (RPE) to derive step labels, and applies consensus filtering (Sec. 3.2).
  - Authors show GenPRM-7B and GenPRM-32B outperform larger classification PRMs on ProcessBench and improve policy BoN/critique refinement (Sec. 4.2–4.3).

 # 8_in_the_flow.pdf

  - Training Set
   - Search-R1 (Jin et al., 2025) — mixed into RL fine-tuning; provides paired question-answer examples for search domain training (Sec. 4.1). (See paper §4.1)
   - DeepMath / Deepmath-103k (He et al., 2025) — mixed into RL fine-tuning for mathematical-domain training (Sec. 4.1). TODO: confirm exact DeepMath variant/name and size in appendix (C.1). (See paper §4.1)

  - Evaluation Set
   - Bamboogle — multi-step compositional reasoning benchmark (Press et al., 2023). (Table 1, Fig. 1, §4.1)
   - 2Wiki (2WikiMultihopQA) — multi-hop QA combining Wikidata + Wikipedia (Ho et al., 2020). (Table 1, §4.1)
   - HotpotQA — multi-hop QA dataset (Yang et al., 2018). (Table 1, §4.1)
   - Musique — multi-step reasoning dataset (Trivedi et al., 2022). (Table 1, §4.1)
   - GAIA — agentic/general assistant benchmark (Mialon et al., 2023). Uses text-only split in experiments. (Table 1, §4.1)
   - AIME24 — AIME 2024 math problems (Art of Problem Solving, 2025). (Table 2, §4.1)
   - AMC23 — American Mathematics Competitions 2023 problems. (Table 2, §4.1)
   - GameOf24 (24-game) — arithmetic puzzle dataset (Lile, 2024). (Table 2, §4.1)
   - GPQA — Graduate-level Google-Proof Q&A benchmark (Rein et al., 2024). (Table 2, §4.1)
   - MedQA — medical exam QA dataset (Di Jin et al., 2021 / LLM-MedQA variants). (Table 2, §4.1)

 Notes:
   - The paper states they mix Search-R1 and DeepMath in RL fine-tuning (Sec. 4.1). It also reports toolset details (Google Search, Wikipedia Search, Web Search, Python Coder, Base Generator) used during rollouts (Sec. 4.1, E.2).
   - TODO: verify exact DeepMath name/variant (paper references Deepmath-103k in refs) and add appendix page refs for dataset mentions (see §C.1 and §C.4).

 # 9_prints.pdf

  - Training Set
   - Annotated preference trajectory-step pairs derived from public agent corpora: Miroverse-v0.1 (MiroMind Data Team, 2025) and Alibaba agent corpora (Wu et al., 2025a; Li et al., 2025b; Tao et al., 2025) — paper states 4,344 information-seeking questions were used to construct annotations, with ~2,294 preference pairs kept after filtering (Sec. A, C). (See A and B)

  - Evaluation Set
   - FRAMES — factual & reasoning-intensive retrieval QA benchmark (Krishna et al., 2025). Paper uses a 300-sample subset (Sec. A).
   - GAIA (Levels 1–3) — general AI assistant benchmark for retrieval + reasoning (Mialon et al., 2024). Paper evaluates on text-only validation split (103 questions) (Sec. A).
   - WebWalkerQA (Easy–Hard) — web traversal QA benchmark requiring web navigation (Wu et al., 2025b). Paper evaluates on 247 English questions (Sec. A).

 Notes:
   - PRINTS training data: annotations generated via Monte Carlo rollouts (M=8) using Qwen3-32B to estimate information-gain scores and summaries; final training set used ~2k preference pairs (Sec. 3.2, A).
   - Evaluation uses LLM-as-Judge (LasJ) with GPT-5 for answer correctness; reported Avg@3 across three runs (Sec. 4.1).
   - The paper also provides implementation/tool details for the Inspect-Eval environment (Serper search API, web-browsing automation, code execution) used in experiments (Sec. A).
