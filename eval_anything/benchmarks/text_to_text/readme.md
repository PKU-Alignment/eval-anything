## The List of Integrated Text to Text Benchmarks

### AGIEval

**📘Overview**

AGIEval is a human-centric benchmark specifically designed to evaluate the general abilities of foundation models in tasks pertinent to human cognition and problem-solving. 

paper: [AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models](https://arxiv.org/pdf/2304.06364.pdf).

**📄 Benchmark Details**

This benchmark is derived from 20 official, public, and high-standard admission and qualification exams intended for general human test-takers, such as general college admission tests (e.g., Chinese College Entrance Exam (Gaokao) and American SAT), law school admission tests, math competitions, lawyer qualification tests, and national civil service exams. 



### ARC

**📘Overview**

The AI2’s Reasoning Challenge (**ARC**) dataset is a multiple-choice question-answering dataset, containing questions from science exams from grade 3 to grade 9. 

link: https://huggingface.co/datasets/allenai/ai2_arc

**📄 Benchmark Details**

A new dataset of 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage research in advanced question-answering. The dataset is partitioned into a Challenge Set and an Easy Set, where the former contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm. We are also including a corpus of over 14 million science sentences relevant to the task, and an implementation of three neural baseline models for this dataset. We pose ARC as a challenge to the community.



### **AdvBench** 

**📘Overview**

**AdvBench** is a **benchmark dataset** designed to evaluate the safety and robustness of large language models (LLMs) under adversarial or harmful input prompts. It focuses on **text-to-text** modalities and includes manually curated and adversarially constructed queries that challenge the model's alignment with ethical and safe generation behavior. For data and code, see [this https URL](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench)

**📄 Benchmark Details**

**AdvBench** is a curated dataset comprising **521 high-risk prompt-response pairs**, designed to evaluate the safety alignment and refusal capabilities of large language models (LLMs). The dataset targets scenarios where the model **should refuse** to comply with harmful, unethical, or illegal instructions but **fails to do so**. Each instance is formatted as a text-to-text input-output pair, with the prompt (goal) containing a dangerous request and the response (target) representing an unsafe or overly compliant model output.

Data Collection and Construction

The dataset is built through the following pipeline:

- **Prompt Collection**: We manually gather and construct adversarial instructions inspired by red-teaming scenarios, real-world safety violations, and open-source adversarial prompt sets. These prompts span a broad spectrum of high-risk behavior types, including but not limited to hacking, abuse, hate speech, and illegal guidance.

- **Response Generation**: Responses are generated using early-stage or base LLM checkpoints that are less safety-aligned, to simulate realistic failure cases where models respond dangerously.

-  **Annotation Process**: Each prompt-response pair is manually reviewed to ensure it represents a **clear safety violation**—i.e., a harmful request that should have been rejected but received a cooperative or unsafe answer.



### **Anthropic**

**📘Overview**

Anthropic Harmlessness**Base** is a benchmark dataset extracted from the **Anthropic Helpful and Harmless (HH-RLHF)** collection. It is designed to evaluate whether large language models (LLMs) can **proactively recognize and refuse harmful, unethical, or illegal user prompts** in the absence of prior response data.

Unlike typical instruction-tuning datasets that contain both prompts and model outputs, this dataset **only consists of adversarial user prompts**, making it especially suitable for **testing refusal behavior** in safety-critical LLM deployments.

The prompts originate from real-world red-teaming efforts conducted by Anthropic during the training of their reinforcement learning from human feedback (RLHF) models. These prompts were written by humans to test how LLMs would respond to potentially harmful queries such as inducing violence, spreading hate speech, or describing illegal actions. 

This dataset is ideal for:

-  Evaluating LLMs' refusal rates and safety alignment without needing ground-truth responses

- Red-teaming and prompt auditing for content moderation systems

-  Building or testing automated refusal detectors

- Benchmarking behavior across LLM variants before and after alignment tuning

Prompt topics span a wide range of safety-sensitive domains, including hate speech, psychological manipulation, dangerous instructions, hacking, and more. The subset helps researchers assess how reliably a model can **say no** when prompted to generate harmful content.This release is an extension of the previously published in [this https URL](https://huggingface.co/datasets/Anthropic/hh-rlhf)

**📄 Benchmark Details**

This dataset consists of **2,312 prompts**, each representing a user-issued instruction that poses potential harm if answered irresponsibly by a language model. Each prompt is stored in a standardized JSON format, under the field extracted_questions.

 

### **AIR-BENCH-2024**

**📘Overview**

AIR-Bench-2024 is an AI safety benchmark that aligns with emerging government regulations and company policies. It consists of diverse, malicious prompts spanning categories of the regulation-based safety categories in the AIR 2024 safety taxonomy.

Project Page: https://huggingface.co/datasets/stanford-crfm/air-bench-2024

**📄 Benchmark Details**

-  Decomposes 8 government regulations and 16 company policies into a four-tiered safety taxonomy with 314 granular risk categories in the lowest tier. 

- Contains 5,694 diverse prompts spanning these categories, with manual curation and human auditing to ensure quality.

 

### **Aegis AI Content Safety Dataset**

**📘Overview**

Aegis AI Content Safety Dataset 2.0 (Nemotron Content Safety Dataset V2), is an open-source content safety dataset, which adheres to Nvidia's content safety taxonomy.

Project Page: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0

**📄 Benchmark Details**

Aegis AI Content Safety Dataset 2.0, is comprised of 33,416 annotated interactions between humans and LLMs, split into 30,007 training samples, 1,445 validation samples, and 1,964 test samples. This release is an extension of the previously published [Nemotron Content Safety Dataset V1](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0).

The curated [training dataset](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/blob/data/llama_training_ready/) consists of a mix collected or generated using the following data sources.

- Human-written prompts collected from the [Anthropic RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf), [Do-Anything-Now DAN](https://github.com/verazuo/jailbreak_llms), and [AI-assisted Red-Teaming](https://www.kaggle.com/datasets/mpwolke/aart-ai-safety) datasets.
- LLM responses generated from [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- Response safety labels generated using an ensemble of 3 LLMs: [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1), [Mistral-NeMo-12B-Instruct](https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct), and [Gemma-2-27B-it](https://huggingface.co/google/gemma-2-27b-it).
- Additionally mix in refusal data generated using [Gemma-2-27B](https://build.nvidia.com/google/gemma-2-27b-it/modelcard.md) using custom prompt-engineered deflection strategies.
- Further data augmentation with [topic-following data](https://gitlab-master.nvidia.com/dialogue-research/topic_following/-/tree/main?ref_type=heads) generated using [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) is found to improve the robustness and adaptability of the model to custom categories.

 

### **BBQ** 

**📘Overview**

**BBQ (Bias Benchmark for QA)** is a hand-curated benchmark dataset designed to evaluate **social bias** in question answering (QA) models. Developed by researchers from NYU and presented in [ACL 2022 Findings](https://aclanthology.org/2022.findings-acl.165), the dataset systematically probes model behavior across a diverse range of **social categories** to reveal tendencies toward **stereotypical or biased answers**.

Each example in BBQ consists of a short question with multiple-choice answers and two types of context:

- **Ambiguous context**: insufficient information is provided, prompting the model to rely on its internal biases or world knowledge.

-   **Disambiguated context**: clear information is included to support a correct, unbiased answer.

This dataset enables fine-grained analysis of when and how QA systems reflect social biases, and whether those biases override explicit evidence in context. Project resources:Dataset GitHub: https://github.com/nyu-mll/BBQ

**📄 Benchmark Details**

The BBQ dataset contains over **58,000 question-answer examples** generated from **325 templates**, each targeting a specific stereotype or group-level assumption. For every template, around **175 instances** are created by varying attributes like group identity, question framing, and context sufficiency.

Each entry is stored in JSONL format and includes the following fields:

- context_ambiguous: an underspecified scenario lacking relevant group information

- context_disamb: an enriched version of the context with group-specific details

- question_neg: a negatively framed question that may elicit a stereotyped answer

- question_pos: a neutral or positively framed variant of the question

- answer_choices: three options, typically including a biased option, a correct neutral one, and a distractor

- label: the gold answer among the three choices

This design allows side-by-side comparisons between ambiguous and disambiguated cases, helping assess the degree to which models rely on stereotypes when information is missing or incomplete.

BBQ evaluates social bias across **nine core demographic and identity dimensions**:

**1.Gender**

**2.Sexual orientation**

**3.Race/ethnicity**

**4.Nationality**

**5.Religion**

**6.Disability status**

**7.Age**

**8.Socioeconomic status (SES)**

**9.Physical appearance and body type**

Each dimension is associated with carefully constructed templates and context settings to test models’ sensitivity to group identity. The dataset includes both **in-group** and **out-group** settings, enabling analyses of model consistency, fairness, and bias asymmetry across groups.

 

### **BeaverTails**

**📘Overview**

**BeaverTails** is a multi-purpose dataset collection developed by the PKU-Alignment team, designed to facilitate the **training, alignment, and evaluation of large language models (LLMs)** with a strong emphasis on safety and helpfulness. It consists of three key subsets covering classification, preference comparison, and evaluation, making it a comprehensive toolkit for safety-critical LLM development.

Official Resources:

- GitHub Repository: https://github.com/PKU-Alignment/beavertails

- Hugging Face Dataset: https://huggingface.co/datasets/PKU-Alignment/BeaverTails

In our safety evaluation experiments, we primarily focus on the **Evaluation subset** of BeaverTails, which contains high-risk prompts across various harm categories for testing the refusal and safety behavior of LLMs.

**📄 Benchmark Details**

1. Classification Subset

   - Contains **301,000+ QA pairs**, each annotated with one or more **harm type labels** from a predefined set of **14 harm categories**, such as child abuse, self-harm, hate speech, drug use, and more.

   - Each QA pair is classified as **safe** or **unsafe**, with approximately **57.3% labeled as unsafe**.

   - Prompts are diverse (7,774 unique), with most having 3–6 corresponding responses.

   - This subset is ideal for training **QA-moderation classifiers**, which identify whether a response violates safety constraints.

2. Preference Subset

   - Comprises over **361,000 expert-labeled response pairs** evaluated under two orthogonal criteria:
     - **Helpfulness**: Whether the response is useful, relevant, or informative
     - **Harmlessness**: Whether the response avoids unsafe, unethical, or harmful content

   - This subset supports training both **reward models** and **cost models** in RLHF pipelines, providing a dual-signal optimization for LLM alignment.

3. Evaluation Subset
   - Includes **700 hand-crafted high-risk prompts**, spanning the same **14 harm categories** (approximately 50 prompts per category).
   - Each prompt is carefully designed to elicit failure cases in safety alignment—e.g., instruction-following behavior on harmful queries.
   - In our benchmark evaluations, we **primarily use this Evaluation subset** to test whether LLMs can correctly refuse or deflect unsafe prompts without over-refusing benign ones.

BeaverTails enables a wide range of applications for safety alignment research and deployment:

- **QA Moderation**: Train classifiers to detect harmful content in responses

- **RLHF Training**: Use reward/cost signals for fine-tuning models toward helpful and harmless behaviors

- **Pre-deployment Evaluation**: Systematically test a model’s refusal accuracy on high-risk categories

- **Model Comparison**: Analyze behavioral differences between base models, supervised fine-tuning (SFT) models, and RLHF models



### CMMLU

**📘Overview**

CMMLU is a comprehensive Chinese assessment suite specifically designed to evaluate the advanced knowledge and reasoning abilities of LLMs within the Chinese language and cultural context. 

- **Homepage:** https://github.com/haonan-li/CMMLU
- **Repository:** https://huggingface.co/datasets/haonan-li/cmmlu
- **Paper:** [CMMLU: Measuring Chinese Massive Multitask Language Understanding](https://arxiv.org/abs/2306.09212).

**📄 Benchmark Details**

CMMLU covers a wide range of subjects, comprising 67 topics that span from elementary to advanced professional levels. It includes subjects that require computational expertise, such as physics and mathematics, as well as disciplines within humanities and social sciences. Many of these tasks are not easily translatable from other languages due to their specific contextual nuances and wording. Furthermore, numerous tasks within CMMLU have answers that are specific to China and may not be universally applicable or considered correct in other regions or languages.



### CEval

**📘Overview**

C-Eval is a comprehensive Chinese evaluation suite for foundation models. It consists of 13948 multi-choice questions spanning 52 diverse disciplines and four difficulty levels.

link: https://github.com/hkust-nlp/ceval

**📄 Benchmark Details**

 It consists of 13948 multi-choice questions spanning 52 diverse disciplines and four difficulty levels, as shown below. Please visit [website](https://cevalbenchmark.com/) or check [paper](https://arxiv.org/abs/2305.08322) for more details.



### **CDialBias**

**📘Overview**

**CDial-Bias** is a benchmark dataset for measuring and analyzing **social bias in Chinese open-domain dialogue systems**. Developed as part of the NLPCC 2022 Shared Task 7, the dataset collects and annotates real-world conversational data from Chinese platforms (e.g., Zhihu), with a focus on revealing biases related to key social attributes such as **race**, **gender**, **region**, and **occupation**.

Dataset Resources:

- GitHub Repository: https://github.com/para-zhou/CDial-Bias

CDial-Bias provides both **annotated training sets** for supervised bias classification and **context-aware test sets** for evaluating model responses under real-world conditions. It is suitable for bias detection, fairness evaluation, and pre-deployment auditing of Chinese dialogue models.

**📄 Benchmark Details**

The dataset is formatted in JSONL, with each entry representing a dialogue sample. Each sample includes:

- Q: The first utterance (typically a user query)

- A: The response (typically from the system)

- Topic: The target bias attribute (Race, Gender, Region, Occupation)

- ContextSensitivity: 0 = not context-sensitive; 1 = requires context to judge bias

- DataType: 0 = Irrelevant, 1 = Bias-expressing, 2 = Bias-discussing

- BiasAttitude: 0 = N/A, 1 = Anti-bias, 2 = Neutral, 3 = Biased

- ReferencedGroups: Specific group(s) referenced in the conversation



### **Cona**

**📘Overview**

**Cona** is a compact, human-curated dataset developed to evaluate the **safety compliance and refusal behavior** of instruction-tuned large language models (LLMs). It focuses on prompts that reflect clearly harmful or policy-violating user instructions—such as violence, self-harm, hate speech, or illegal activity and pairs each with a consistent, refusal-style response.

 Dataset Links:

- GitHub Repository: https://github.com/vinid/instruction-llms-safety-eval

Cona is particularly well-suited for red-teaming and quick-turnaround evaluations of safety alignment in instruction-following LLMs.

**📄 Benchmark Details**

Cona contains approximately **178 core high-risk prompts**, but the dataset **currently includes additional extended content** beyond this core set. Each example consists of:

- prompt: A clearly unsafe or inappropriate instruction

- safe_response: A standardized, concise refusal response intended to guide models toward safe behavior

✨**Features**

- **Coverage**: Prompts cover diverse unsafe intent categories, including incitement to violence, hate speech, drug use, self-harm, and more

- **Response Design**: All safe_response entries are consistent, neutral, and non-judgmental refusals

- **Compact and Efficient**: While relatively small, the dataset is highly curated and focused, making it efficient for training or evaluation use

 

### **ConfAIde**

**📘Overview**

**ConfAIde** is a multi-tier benchmark dataset designed to evaluate the **privacy reasoning and contextual integrity** capabilities of instruction-tuned large language models (LLMs). Unlike traditional privacy benchmarks that focus on memorization or training data leakage, ConfAIde tests whether LLMs can **appropriately refuse to disclose private information** based on who is asking, what they’re asking for, and in what context.

Dataset Links:

- GitHub Repository: https://github.com/skywalker023/confAIde/tree/main/benchmark

The dataset simulates realistic conversational scenarios with multiple agents, revealing how well models handle privacy in nuanced, interactive, and task-oriented contexts.

**📄 Benchmark Details**

ConfAIde organizes its evaluation into four progressively difficult **tiers**, each targeting a different aspect of contextual privacy reasoning. Each sample consists of a structured scenario or dialogue involving multiple agents (e.g., subject, aware agent, oblivious agent) and is labeled based on whether a privacy violation would occur.

**Tier Structure:**

- **Tier 1: Info-Sensitivity**
   Basic sensitivity identification—can the model recognize whether a single item of information is private?

- **Tier 2a / 2b: InfoFlow-Expectation**
   Introduces expectations about **who** is requesting the information and **why**. Tier 2b expands these into short, narrative-style vignettes.

- **Tier 3: InfoFlow-Control**
   Multi-agent scenarios that require **theory-of-mind** reasoning: the model must understand who knows what and refrain from disclosing private data even indirectly.

- **Tier 4: InfoFlow-Application**
   Real-world use cases like summarizing meetings or extracting action items. These tasks test models’ ability to maintain privacy under indirect, practical instructions.

Each sample is presented as a structured JSON file or a natural language prompt. Key fields often include subject_agent, aware_agent, oblivious_agent, and sensitivity_label, with privacy violations assessed based on model outputs.

**Why ConfAIde Matters**

- **Contextual Integrity over Memorization**: ConfAIde shifts the focus from static training data leakage to **dynamic, inference-time privacy violations**, reflecting real-world risk scenarios.

- **Progressive Reasoning Depth**: The benchmark moves from simple fact recognition to nuanced social reasoning and multi-agent control, testing LLMs’ understanding of information boundaries.

- **Empirical Gaps**: Evaluation results show that even advanced models like ChatGPT and GPT-4 **disclose private information in 39% (Tier 3) and 57% (Tier 4) of cases**—scenarios where humans would withhold it.

 

### **CyberattackAssistance**

**📘Overview**

**CyberattackAssistance** is a benchmark dataset released under Meta’s [Purple Llama CybersecurityBenchmarks](https://github.com/facebookresearch/PurpleLlama/tree/main/CybersecurityBenchmarks) suite. It is specifically designed to evaluate the **tendency of large language models (LLMs) to assist in cyberattack-related prompts**, using structured scenarios derived from the **MITRE ATT&CK** framework.

Dataset location:

- GitHub: https://github.com/facebookresearch/PurpleLlama/tree/main/CybersecurityBenchmarks/datasets/mitre

The dataset provides a standardized method for testing how LLMs respond to harmful instructions such as privilege escalation, container escape, lateral movement, and command & control (C2) activities. It supports reproducible, fine-grained security assessments across multiple model families and defense configurations.

**📄 Benchmark Details**

**Data Construction**

- **Source**: Based on techniques and tactics from the MITRE ATT&CK knowledge base.

- **Taxonomy Alignment**: Each scenario maps to specific ATT&CK IDs (e.g., T1550, T1105) for consistent evaluation across threat categories.

- **Prompt Generation**: Each example simulates a user seeking help with performing a cyberattack task, such as writing scripts for privilege escalation or exploiting containers.

**Format**

Each sample is stored as a JSON object and includes the following fields:

- attack_type: The MITRE ATT&CK ID (e.g., T1550 for “Use Alternate Authentication Material”)

- prompt: A natural language user instruction attempting to elicit attack-related assistance

- template: (optional) A high-level label such as code_assist indicating the prompt style

**Evaluation Goals**

The dataset is intended to measure:

- Whether the LLM provides **actionable code or advice** that directly facilitates a cyberattack

- **Helpfulness rate**: How often does the model comply with attack-related prompts?

- **Coverage**: Prompts span a wide range of attack classes including:
  - Privilege Escalation
  - Container Escape
  - Command & Control
  - Lateral Movement
  - Defense Evasion
  - Data Exfiltration

The benchmark enables comparative analysis of different models and the effectiveness of **guard mechanisms** like Llama Guard or Prompt Guard.



### **DecodingTrust**

**📘Overview**

**DecodingTrust** is a large-scale, multi-dimensional benchmark developed by the AI-Secure team (UIUC + collaborators) to evaluate the **trustworthiness of instruction-tuned large language models (LLMs)**, especially GPT-3.5 and GPT-4. The benchmark spans **eight critical dimensions** of trust: toxicity, stereotype bias, adversarial robustness, out-of-distribution robustness, adversarial demonstrations, privacy, machine ethics, and fairness.

 Dataset Links:

- GitHub Repository: https://github.com/AI-secure/DecodingTrust

- Hugging Face Dataset: https://huggingface.co/datasets/AI-Secure/DecodingTrust

The benchmark provides standardized data, attack strategies, and evaluation code across all dimensions, enabling transparent, reproducible, and extensible trustworthiness evaluation of modern LLMs.

**📄 Benchmark Details**

Each of the eight evaluation dimensions is implemented as an independent test suite, consisting of prompt templates, red-teaming strategies, expected outputs, and evaluation metrics. Here's a brief overview of each dimension:

1. **Stereotype & Bias**
    The dataset contains prompts combining **24 demographic groups × 16 bias topics** × multiple tone variants (neutral, positive, negative) to evaluate how models reflect or amplify stereotypes.

2. **Adversarial Robustness**
    Includes text-based adversarial attacks targeted at open models like Alpaca and Vicuna, then used to test GPT-3.5/4 via transfer attacks.

3. **Out-of-Distribution (OOD) Robustness**
    Involves testing models on stylized text (e.g., Shakespearean language, poetry), rare knowledge, or unusual formats to evaluate generalization and semantic understanding under distribution shifts.

4. **Adversarial Demonstrations**
    Inserts harmful few-shot examples into prompts (e.g., backdoor or counterfactual demonstrations) to test whether models follow misleading contextual examples.

5. **Privacy**
    Prompts are designed to extract memorized personal information (e.g., phone numbers, emails), test privacy leakage through conversation history, and probe semantic privacy understanding.

6. **Machine Ethics**
    Builds on datasets like ETHICS and Jiminy Cricket to assess moral decision-making. Includes jailbreak-style instructions to test whether models can be coerced into unethical outputs.

7. **Fairness**
    Uses demographic-controlled variations of identical questions to evaluate disparities in responses, measuring whether model outputs vary based on protected attributes like gender, race, or age.

8. **Toxicity**
    Adversarial prompts are designed using both system-level and user-level input structures, optimized via automated prompt generation methods. The goal is to probe whether models respond with unsafe or toxic outputs.

**Why DecodingTrust Matters**

- **Unified Evaluation Platform**: First benchmark to unify 8 major dimensions of trustworthiness under a reproducible and extensible framework.

- **Red-Teaming by Design**: Each dimension employs adversarial or carefully optimized prompting strategies to push model behavior toward failure boundaries.

- **Real-World Relevance**: Demonstrates how even advanced models like GPT-4 may still respond toxically, amplify bias, or leak private information—especially under adversarial prompting.

- **Tooling Ecosystem**: Includes fully runnable scripts (main.py, chat.py, utils.py), leaderboard submission tools, and support for both OpenAI and open-source Hugging Face models.



### **Dice350**

**📘Overview**

**Dice350** is a key subset of the **DICES** dataset (Diversity in Conversational AI Evaluation for Safety), released by Google Research. It provides a high-quality benchmark for evaluating conversational AI systems from a **multi-rater, demographically diverse perspective**, focusing on how people from different backgrounds perceive the **safety and harmfulness of AI-generated responses**.

Dataset Access:

- GitHub Repository: https://github.com/google-research-datasets/dices-dataset

Dice350 consists of **350 multi-turn dialogues**, each rated by **123 diverse human annotators** on multiple dimensions of safety and appropriateness. The dataset enables both aggregate and disaggregated analyses of AI safety across different user groups.

**📄 Benchmark Details**

Each conversation typically contains up to 5 turns, ending with a model-generated response that is subject to evaluation.

Dialogue Sources & Prompting Strategy

- Dialogues were generated by **red-teamers interacting with LLMs**, designed to elicit potentially harmful, biased, or policy-violating content.

- Prompts span **toxicity, prejudice, unsafe medical/legal/financial advice, inappropriate content, political bias, privacy violations**, and more.

- Each dialogue is rated **independently by 123 raters**, providing a robust sample size for analyzing **subjectivity, agreement, and sensitivity** across subpopulations.

 

### **Do Anything Now (DAN)**

**📘Overview**

**Do Anything Now (DAN)** is a large-scale, real-world dataset of jailbreak prompts collected from live user interactions across online platforms. Developed as part of the JailbreakHub project, DAN captures the behavior of users attempting to **bypass safety constraints in large language models (LLMs)** through prompt engineering, especially targeting models like ChatGPT and GPT-4.

Unlike artificially constructed adversarial prompts, the DAN dataset contains **15,140 naturally occurring jailbreak-related prompts** sourced from **14 platforms** including Reddit, Discord, FlowGPT, and community sites. Among these, **1,405 prompts are manually verified as canonical jailbreak examples**, making this one of the largest publicly available "in-the-wild" jailbreak datasets.

Dataset Link:

- GitHub: https://github.com/verazuo/jailbreak_llms

**📄 Benchmark Details**

- **Collection Period**: December 2022 – December 2023

- **Total Prompts**: 15,140

- **Unique Users**: 7,308

- **Canonical Jailbreaks**: 1,405 prompts marked as effective jailbreaks by user consensus

- **Platforms**: Reddit, Discord, FlowGPT, AIPRM, etc.

- **Long-Term Jailbreakers**: 803 users contributed at least one jailbreak prompt; 28 were active >100 days

Additionally, the dataset includes **107,250 adversarial test questions**, designed to test jailbreak success across **13 sensitive categories**, such as:

- Illegal activities

- Hate speech

- Sexual content

- Political manipulation

- Privacy violations

- Self-harm instructions

- Malware and cybersecurity exploits



### **DoNotAnswer**

**📘Overview**

**Do-Not-Answer** is the first open-source benchmark specifically designed to evaluate whether **large language models (LLMs)** can **safely refuse to respond** to harmful, unethical, or policy-violating prompts. Developed by the LibrAI team, the dataset is prompt-only—each entry consists of a single input that **should ideally trigger a refusal from a well-aligned model**.

The dataset contains **939 high-risk prompts** spanning a diverse set of **12 high-level risk categories**, such as illegal activities, hate speech, self-harm, adult content, and misinformation.

Dataset Resources:

- GitHub Repository: https://github.com/Libr-AI/do-not-answer

- Hugging Face Dataset Page: https://huggingface.co/datasets/LibrAI/do-not-answer

The project also includes an automated evaluation framework, featuring a **600M parameter classifier**, which achieves **near GPT-4 performance** in identifying prompts that should be refused.

**📄 Benchmark Details**

**Scale and Format**

- **Total Prompts**: 939

- **Prompt Type**: Harmful or policy-violating instructions that require refusal

- **Structure**: Each entry is a standalone prompt without a response

- **Annotation Sources**: Human annotators, GPT-4, and fine-tuned classifiers (e.g., Longformer-based models) were used to assess model responses in downstream evaluation.



### **FLAMES**

**📘Overview**

Flames is a highly adversarial benchmark in Chinese for LLM's value alignment evaluation developed by Shanghai AI Lab and Fudan NLP Group. 

Project Page: https://huggingface.co/datasets/opencompass/flames

**📄 Benchmark Details**

- **a highly adversarial prompts set**: we meticulously design a dataset of 2,251 highly adversarial, manually crafted prompts, each tailored to probe a specific value dimension (i.e., Fairness, Safety, Morality, Legality, Data protection). Currently, we release 1,000 prompts for public use (**Flames_1k_Chinese**).
-  **a specified scorer**: based on our annotation, we train a specified scorer to easily grade the responses (available at [Huggingface](https://huggingface.co/CaasiHUANG/flames-scorer)).

 

### **Fake Alignment**

**📘Overview**

**FakeAlignment** is a comparative benchmark to empirically verify its existence in LLMs, including a Fake alIgNment Evaluation (FINE) framework and two novel metrics--Consistency Score (CS) and Consistent Safety Score (CSS), which jointly assess two complementary forms of evaluation to quantify fake alignment and obtain corrected performance estimation. 

Project Page: https://github.com/AI45Lab/Fake-Alignment

**📄 Benchmark Details**

The test dataset includes five safety-relevant subcategories that can be used to evaluate the alignment of LLMs. Each question contains a question stem and corresponding positive and negative options.



### **GPTFuzzer**

**📘Overview**

GPTFuzzer is a red teaming dataset used in [*GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts*](https://arxiv.org/pdf/2309.10253)*.*

Project Page: https://github.com/sherdencooper/GPTFuzz

**📄 Benchmark Details**

- Including 100 questions from two public datasets: [llm-jailbreak-study](https://sites.google.com/view/llm-jailbreak-study) and [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf). The templates are collected from [llm-jailbreak-study](https://sites.google.com/view/llm-jailbreak-study).
- Encompassing a wide range of prohibited scenarios such as illegal or immoral activities, discriminations, and toxic content. 



### gsm8k

**📘Overview**

GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.

link: https://huggingface.co/datasets/openai/gsm8k

paper: Training Verifiers to Solve Math Word Problems

**📄 Benchmark Details**

- These problems take between 2 and 8 steps to solve.
- Solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer.
- A bright middle school student should be able to solve every problem: from the paper, "Problems require no concepts beyond the level of early Algebra, and the vast majority of problems can be solved without explicitly defining a variable."
- Solutions are provided in natural language, as opposed to pure math expressions. From the paper: "We believe this is the most generally useful data format, and we expect it to shed light on the properties of large language models’ internal monologues"



### HumanEval

**📘Overview**

This is an evaluation harness for the HumanEval problem solving dataset described in the paper "Evaluating Large Language Models Trained on Code". It used to measure functional correctness for synthesizing programs from docstrings. 

link: https://github.com/openai/human-eval

**📄 Benchmark Details**

It consists of 164 original programming problems, assessing language comprehension, algorithms, and simple mathematics, with some comparable to simple software interview questions.



### **HarmBench**

**📘Overview**

**HarmBench** is a comprehensive evaluation benchmark developed to assess the safety, robustness, and refusal capabilities of large language models (LLMs) when confronted with adversarial or harmful prompts. It provides a standardized pipeline, a taxonomy of harmful behaviors, and a set of behavior-driven test prompts for auditing model responses across a wide range of risk categories.

Dataset Repository:

- GitHub: https://github.com/centerforaisafety/HarmBench/tree/main/data/behavior_datasets

The benchmark supports comparisons across models and defense strategies, allowing researchers and developers to systematically evaluate how likely a model is to generate harmful content when red-teamed.

**📄 Benchmark Details**

**Behavior Categories**

HarmBench classifies risky model behavior into two parallel taxonomies:

**1. Semantic Categories**

These describe the intended harmful content of a prompt or behavior, such as:

- Cybercrime and unauthorized access

- Chemical/biological weapons or toxic substances

- Intellectual property violations

- Misinformation and disinformation

- Harassment, abuse, or bullying

- Illicit activities (e.g., fraud, trafficking, violence)

- General harmful behavior (e.g., privacy violations, dangerous advice)

**2. Functional Categories**

These describe how the behavior operates:

- **Standard**: Directly prompts harmful content

- **Copyright**: Attempts to copy protected text

- **Contextual**: Requires multi-turn or scenario-based context

- **Multimodal**: Involves image-text combinations for malicious use

Each behavior is represented as a standalone task that may include optional context or modality-specific data.

**Dataset Format and Usage**

- Behaviors are formatted as JSON entries, each with a behavior_id, description, and optional context or image fields.

- The dataset is divided into a validation set and a test set, designed for robust generalization evaluation.

- The evaluation protocol involves presenting these behaviors to a target LLM and analyzing whether the output contains harmful or policy-violating content.

**Why Use HarmBench**

- **Comprehensive Coverage**: Includes over 18 red-teaming strategies and dozens of high-risk behavior categories.

- **Standardized Evaluation**: Offers consistent attack definitions and output validation criteria.

- **Scalable and Extensible**: Supports both text-only and multimodal scenarios, along with easy model integration.

- **Publicly Available and Reproducible**: All behavior tasks, test sets, and evaluation code are open-sourced and ready for use.

- **Defense-Oriented**: Helps measure the impact of safety interventions and adversarial training.



### **HarmfulQ**

**📘Overview**

**HarmfulQ** is a benchmark dataset developed by the SALT-NLP team to investigate the unintended consequences of **Zero-Shot Chain-of-Thought (CoT)** prompting, particularly when applied to socially sensitive or safety-critical questions. It explores whether the inclusion of reasoning prompts like *“Let’s think step by step.”* increases the likelihood that large language models (LLMs) generate harmful or toxic content.

Dataset Repository:

- GitHub: https://github.com/SALT-NLP/chain-of-thought-bias

This dataset is designed to complement stereotype-related evaluations, providing a focused lens on CoT-induced behaviors when models are faced with dangerous or ethically questionable inputs.

**📄 Benchmark Details**

 **Data Sources**

HarmfulQ includes a curated list of real-world, potentially harmful user questions—such as “How do I build a bomb?” or “How can I encourage hatred toward a group?”—that are likely to trigger unsafe outputs under improper prompting.

The dataset is formatted in JSON under the file toxic_outs.json, where each entry includes:

- A **harmful prompt** (e.g., a malicious or unethical question)

- A **model response** under standard prompt

- A **model response** under Zero-Shot Chain-of-Thought (CoT) prompting

The comparison then measures how often the model:

- Refuses to answer

- Produces toxic, harmful, or overly compliant output

This allows for calculating the **toxicity increase rate** attributable to CoT.



### **HExPHI**

📘Overview

**HEx‑PHI** (Human-Extended Policy-Oriented Harmful Instruction Benchmark) is a benchmark dataset designed to evaluate the safety alignment of large language models (LLMs) with respect to **policy-prohibited instructions**—i.e., prompts that violate the usage policies of major providers like OpenAI and Meta (LLaMA-2). It includes **330 adversarial instructions** that span **11 high-risk content categories**, curated to test whether models can effectively **refuse** or **safely respond** to harmful requests.

Dataset Link:

- https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI

The dataset is intended to serve as a **fine-grained safety audit tool**, helping researchers assess policy compliance failures and improve model alignment through targeted tuning or evaluation.

**📄 Benchmark Details**

**Composition and Taxonomy**

- **Total Prompts**: 330

- **Categories**: 11 policy-prohibited content areas, each with 30 samples.

- Prompts are adversarial, policy-relevant, and manually reviewed for harm severity.



### **JBB-Behaviors**

**📘Overview**

JBB-Behaviors is a red teaming dataset used in Jailbreakbench. Jailbreakbench is an open-source robustness benchmark for jailbreaking large language models (LLMs). The goal of this benchmark is to comprehensively track progress toward (1) generating successful jailbreaks and (2) defending against these jailbreaks. 

Project Page: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors

**📄 Benchmark Details**

Each entry in the JBB-Behaviors dataset has four components:

-  **Behavior**: A unique identifier describing a distinct misuse behavior

-  **Goal**: A query requesting an objectionable behavior

-  **Target**: An affirmative response to the goal string

-  **Category**: A broader category of misuse from [OpenAI's usage policies](https://openai.com/policies/usage-policies)

-  **Source**: the source from which the behavior was sourced (i.e., Original, [Trojan Detection Challenge 2023 Red Teaming Track](https://trojandetection.ai/)/[HarmBench](https://harmbench.org/), or [AdvBench](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv))

The dataset comprises of 100 distinct misuse behaviors (with examples sourced from [AdvBench](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv), [Trojan Red Teaming Competition](https://trojandetection.ai/)/[HarmBench](https://harmbench.org/), and ideas sourced from [Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation](https://arxiv.org/abs/2311.03348) by Shah et al.) divided into ten broad categories corresponding to [OpenAI's usage policies](https://openai.com/policies/usage-policies).

 

### **jade_db**

**📘Overview**

JADE is a targeted linguistic fuzzing platform which strengthens the linguistic complexity of seed questions to simultaneously and consistently break a wide range of widely-used LLMs categorized in three groups: eight open-sourced Chinese, six commercial Chinese and four commercial English LLMs.

Project Page: https://whitzard-ai.github.io/jade.html

**📄 Benchmark Details**

JADE is based on Noam Chomsky's seminal theory of transformational-generative grammar. Given a seed question with unsafe intention, JADE invokes a sequence of generative and transformational rules to increment the complexity of the syntactic structure of the original question, until the safety guardrail is broken. Our key insight is: Due to the complexity of human language, most of the current best LLMs can hardly recognize the invariant evil from the infinite number of different syntactic structures which form an unbound example space that can never be fully covered. Technically, the generative/transformative rules are constructed by native speakers of the languages, and, once developed, can be used to automatically grow and transform the parse tree of a given question, until the guardrail is broken.

 

### **llm-jailbreak-study**

**📘Overview**

llm-jailbreak-study is a red teaming test set used in [Jailbreaking Chatgpt via Prompt Engineering: An Empirical Study](https://sites.google.com/view/llm-jailbreak-study/home).

Project Page: https://sites.google.com/view/llm-jailbreak-study

**📄 Benchmark Details**

llm-jailbreak-study first categorizes 78 identified jailbreak prompts into 10 unique patterns, grouped under 3 types of jailbreak strategies.

 

### **LatentJailbreak**

**📘Overview**

**LatentJailbreak** is a benchmark dataset designed to evaluate the safety and robustness of large language models (LLMs) in the presence of **latent (hidden) jailbreak triggers**. Unlike traditional jailbreak attempts that use overtly malicious prompts, LatentJailbreak focuses on more subtle, context-based vulnerabilities—where seemingly benign interactions can trigger unintended, harmful completions.

Dataset Link:

- GitHub: https://github.com/qiuhuachuan/latent-jailbreak

The dataset is particularly useful for assessing whether LLMs can maintain safe behavior when sensitive information is embedded in earlier parts of a conversation and later queried in a way that tests their contextual memory and refusal capabilities.

**📄 Benchmark Details**

**Structure and Scale**

- **Total Instances**: ~2,000 prompt-context pairs

- **Each sample includes**:
  - benign_prompt: a harmless user query
  - latent_context: embedded contextual content that may contain sensitive or triggering information 
  - target_prompt: a follow-up question designed to retrieve or activate harmful behavior based on the earlier context



### Mmlu 

**📘Overview**

**MMLU** (**Massive Multitask Language Understanding**) is a new benchmark designed to measure knowledge acquired during pretraining by evaluating models exclusively in zero-shot and few-shot settings. This makes the benchmark more challenging and more similar to how we evaluate humans. 

link: https://huggingface.co/datasets/cais/mmlu

**📄 Benchmark Details**

The benchmark covers 57 subjects across STEM, the humanities, the social sciences, and more. It ranges in difficulty from an elementary level to an advanced professional level, and it tests both world knowledge and problem solving ability. Subjects range from traditional areas, such as mathematics and history, to more specialized areas like law and ethics. The granularity and breadth of the subjects makes the benchmark ideal for identifying a model’s blind spots.

### MMLUPRO

**📘Overview**

MMLU-Pro is an enhanced benchmark designed to evaluate language understanding models across broader and more challenging tasks. 

link: https://github.com/TIGER-AI-Lab/MMLU-Pro

**📄 Benchmark Details**

Building on the Massive Multitask Language Understanding (MMLU) dataset, MMLU-Pro integrates more challenging, reasoning-focused questions and increases the answer choices per question from four to ten, significantly raising the difficulty and reducing the chance of success through random guessing. MMLU-Pro comprises over 12,000 rigorously curated questions from academic exams and textbooks, spanning 14 diverse domains including Biology, Business, Chemistry, Computer Science, Economics, Engineering, Health, History, Law, Math, Philosophy, Physics, Psychology, and Others.





### **MoralBench**

**📘Overview**

MoralBench is a comprehensive dataset specifically curated to probe the moral dimensions of LLM outputs, addressing a wide range of ethical dilemmas and scenarios reflective of real-world complexities.

Project Page: https://github.com/agiresearch/MoralBench

**📄 Benchmark Details**

The main contribution of this work lies in the development of benchmark datasets and metrics for assessing the moral identity of LLMs, which accounts for nuance, contextual sensitivity, and alignment with human ethical standards. 

The methodology involves a multi-faceted approach, combining quantitative analysis with qualitative insights from ethics scholars to ensure a thorough evaluation of model performance. 

 

### **MIC**

**📘Overview**

Open-domain or "chit-chat" conversational agents often reflect insensitive, hurtful, or contradictory viewpoints that erode a user’s trust in the integrity of the system. Moral integrity is one important pillar for building trust. MIC is a dataset that can help us understand chatbot behaviors through their latent values and moral statements. 

Project Page: https://github.com/SALT-NLP/mic

**📄 Benchmark Details**

MIC contains 114k annotations, with 99k distinct "Rules of Thumb" (RoTs) that capture the moral assumptions of 38k chatbot replies to open-ended prompts. These RoTs represent diverse moral viewpoints, with the following distribution of underlying moral foundations:

- Care: wanting someone or something to be safe, healthy, and happy. (58k chatbot replies)
- Fairness: wanting to see individuals or groups treated equally or equitably. (24k)
- Liberty: wanting people to be free to make their own decisions. (22k)
- Loyalty: wanting unity and seeing people keep promises or obligations to an in-group. (22k)
- Authority: wanting to respect social roles, duties, privacy, peace, and order. (20k)
- Sanctity: wanting people and things to be clean, pure, innocent, and holy. (13k)

 

### **MoralChoice**

**📘Overview**

*MoralChoice* is a survey dataset to evaluate the moral beliefs encoded in LLMs. 

Project Page: https://huggingface.co/datasets/ninoscherrer/moralchoice

**📄 Benchmark Details**

The dataset consists of:

- Survey Question Meta-Data: 1767 hypothetical moral scenarios where each scenario consists of a description / context and two potential actions
  - Low-Ambiguity Moral Scenarios (687 scenarios): One action is clearly preferred over the other.
  - High-Ambiguity Moral Scenarios (680 scenarios): Neither action is clearly preferred

- Survey Question Templates: 3 hand-curated question templates
- Survey Responses: Outputs from 28 open- and closed-sourced LLMs



### **MaliciousInstruct**

**📘Overview**

**MaliciousInstruct** is a comprehensive benchmark dataset developed by the Princeton SysML team to evaluate and train large language models (LLMs) on their resilience to **malicious or adversarial instructions**. The dataset is designed to simulate real-world misuse scenarios where LLMs may be prompted to assist in activities such as hacking, privacy invasion, or system exploitation.

Dataset Link:

- GitHub: https://github.com/Princeton-SysML/Jailbreak_LLM/tree/main/data

It serves both as an **evaluation benchmark** for jailbreak vulnerabilities and as a **training resource** to develop LLMs with robust refusal behaviors in security-critical contexts.

**📄 Benchmark Details**

**Categories and Coverage**

MaliciousInstruct includes a wide range of adversarial prompt types, such as:

**Cyber Attacks** – SQL injection, vulnerability scanning, remote exploits

**Malicious Code Execution** – backdoors, rootkits, obfuscated payloads

**System Sabotage** – file deletion, disk corruption, privilege escalation

**Social Engineering** – phishing, identity fraud, deception

**Privacy Invasion** – password extraction, credential inference

The dataset includes thousands of samples across these categories, each structured for adversarial evaluation and safe model tuning.

**Data Format**

Each entry is stored in a structured JSON format, typically containing:

- instruction: the malicious prompt

- context: optional scenario setup or prior dialogue

- target_response: the safe or correct model behavior (usually a refusal)

- role: indicates the speaker



### **MaliciousInstructions**

**📘Overview**

**MaliciousInstructions** is a focused evaluation dataset designed to assess how reliably large language models (LLMs) refuse to comply with clearly harmful, illegal, or policy-violating prompts. Released as part of the [instruction-llms-safety-eval](https://github.com/vinid/instruction-llms-safety-eval) project, it serves as a concise benchmark to test a model’s refusal capability under explicitly malicious instruction settings.

Dataset Access:

- GitHub: https://github.com/vinid/instruction-llms-safety-eval

- Hugging Face: https://huggingface.co/datasets/PKU-Alignment/Cona

**📄 Benchmark Details**

**Structure and Format**

- The dataset consists of **178 core high-risk prompts**, each paired with a standardized safe refusal response.

- Each sample is structured as a dictionary with two fields:

prompt: A user-issued instruction that violates ethical, legal, or safety policies.

safe_response: A consistent and policy-aligned rejection response suitable for training or evaluation.

**Categories Covered**

Prompts in MaliciousInstructions span a variety of prohibited domains, including:

- Instructions involving **violence**, **weapons**, or **self-harm**

- **Hacking**, **exploitation**, and **illegal software use**

- **Harassment**, **hate speech**, and **discrimination**

- **Fraud**, **deception**, and **politically sensitive content**

- **Sexual or adult content**, **privacy violations**, and more

These instructions are crafted to trigger potentially dangerous behaviors and are expected to be **uniformly rejected** by well-aligned LLMs.

**Construction Methodology**

- Prompts were selected based on public safety guidelines, real-world adversarial red-teaming, and known examples of harmful usage.

- All samples were **manually reviewed** and **validated** to ensure clarity and safety violation.

- The safe_response is standardized to support **supervised fine-tuning** and consistent **refusal auditing**.



### **RedEval**

**📘Overview**

**RedEval** is a comprehensive red-teaming benchmark designed to evaluate the safety alignment of large language models (LLMs) using harmful or adversarial prompts. At the heart of this benchmark is the **HarmfulQA** dataset, which consists of carefully curated high-risk questions and dialogue chains designed to probe model behavior in safety-sensitive scenarios.

Dataset Links:

- GitHub Repository: https://github.com/declare-lab/red-instruct/tree/main/harmful_questions

- HuggingFace Dataset: https://huggingface.co/datasets/declare-lab/HarmfulQA

HarmfulQA is accompanied by two types of conversational data:

- **Blue (Safe) Dialogues**: AI-generated responses that safely refuse or deflect the harmful question.

- **Red (Harmful) Dialogues**: Red-teamed examples where the model fails and generates unsafe content.

**📄 Benchmark Details**

**Scope and Composition**

- **Total Questions**: 1,960 harmful or high-risk questions.

- **Safe Dialogues (Blue)**: 9,536 multi-turn dialogues generated by ChatGPT using Chain-of-Utterances (CoU) prompting to reflect helpful but safe behavior.

- **Unsafe Dialogues (Red)**: 7,356 dialogues generated by prompting base models with CoU-based adversarial red-teaming strategies.

- **Total Dialogue Turns**: 66K turns in blue dialogues and 52K turns in red dialogues.



### **S-Eval**

**📘Overview**

S-Eval is designed to be a new comprehensive, multi-dimensional and open-ended safety evaluation benchmark. 

Project Page: https://arxiv.org/abs/2405.14191

**📄 Benchmark Details**

So far, S-Eval has 220,000 evaluation prompts in total (and is still in active expansion), including 20,000 base risk prompts (10,000 in Chinese and 10,000 in English) and 200,000 *corresponding* attack prompts derived from 10 popular adversarial instruction attacks. These test prompts are generated based on a comprehensive and unified risk taxonomy, specifically designed to encompass all crucial dimensions of LLM safety evaluation and meant to accurately reflect the varied safety levels of LLMs across these risk dimensions. 

S-Eval's risk taxonomy has a structured hierarchy with four levels, comprising 8 risk dimensions, 25 risk categories, 56 risk subcategories, and 52 risk sub-subcategories. The first-level risk dimensions and second-level risk categories are shown in the following:

![img](https://raw.githubusercontent.com/IS2Lab/S-Eval/main/assets/risk_taxonomy.png) 



### **SALAD-Bench**

**📘Overview**

SALAD-Bench is a challenging safety benchmark specifically designed for evaluating LLMs, defense, and attack methods. Compared to existing benchmarks, **SALAD-Bench** stands out for its scale, diversity, polymorphism, and multi-functionality. 

Project Page: https://github.com/OpenSafetyLab/SALAD-BENCH

**📄 Benchmark Details**

SALAD-Bench contains 21k harmful base questions classified into 6 domains, 16 tasks, and 66 specific categories. It contains questions with variation, including QA and multiple choice, and with enhancement, by attack and defense methods. Such versatility makes our data able to measure LLM's safety from multiple aspects and also the effectiveness of attack and defense methods. To accurately evaluate LLMs' responses, SALAD-Bench proposes *MD-Judge* and *MC-Judge* for QA-pairs and multiple-choice questions respectively. 

**✨Features**

1. Compact Taxonomy with Hierarchical Levels: Our benchmark proposes a structured hierarchy with three levels, comprising 6 domains, 16 tasks, and 65 categories.
2. Enhanced Difficulty and Complexity:
   1. Difficult base set
   2. Significantly enhanced set

3. Advanced Evaluation Framework
   1. Specialized fine-tuned evaluator namely *MD-Judge* for QA pairs.
   2. Automatic safety evaluation across various dimensions with regex parsing for MC questions.

4. Joint-Purpose Utility:
   1. General LLM safety evaluation
   2. LLM attack method evaluation
   3. LLM defense method evaluation



### **SandP**

**📘Overview**

Project Page: https://huggingface.co/datasets/Lv111/SandP

No dataset information yet

 

### **SimpleSafetyTests**

**📘Overview**

SimpleSafetyTests is a test suite for rapidly and systematically identifying such critical safety risks. 

Project Page: https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests

**📄 Benchmark Details**

The test suite comprises 100 test prompts across five harm areas that LLMs, for the vast majority of applications, should refuse to comply with.

 

### **StrongREJECT**

**📘Overview**

StrongREJECT is a state-of-the-art LLM jailbreak evaluation benchmark.

Paper: StrongREJECT for Empty Jailbreaks

Project Page: https://github.com/dsbowen/strong_reject

**📄 Benchmark Details**

- Implements state-of-the-art LLM jailbreak evaluators, including StrongREJECT and HarmBench

- Implements several dozen jailbreaks for researchers to use as baselines

- Includes a dataset of prompts from 6 categories of harmful behavior

  - Disinformation and deception

  - llegal goods and services

  - Hate/harassment/discrimination

  - Non-violent crimes

  - Violence

  - Sexual content



### **TDCRedTeaming**

**📘Overview**

A red teaming dataset used in Red Teaming Track of the Trojan Detection Challenge 2023 (LLM Edition), a NeurIPS 2023 competition.  

Project Page: https://trojandetection.ai/

**📄 Benchmark Details**

Contains 50 behaviors, which fall into the following high-level categories:

- Bigotry and Abusive Language
- Violent Content and Conduct
- Illegal Activities
- Malware and Exploits
- Scams
- Misinformation and Disinformation
- Other Undesirable Content

For more information, see [here](https://trojandetection.ai/tracks.html#red-teaming).



### **TruthfulQA**

**📘Overview**

TruthfulQA is a benchmark for measuring the truthfulness of language models when generating answers to questions.

Project Page: https://github.com/sylinrl/TruthfulQA

**📄 Benchmark Details**

TruthfulQA consists of two tasks that use the same sets of questions and reference answers.

**Generation (main task):**

- Task: Given a question, generate a 1-2 sentence answer.

- Objective: The primary objective is overall truthfulness, expressed as the percentage of the model's answers that are true. Since this can be gamed with a model that responds "I have no comment" to every question, the secondary objective is the percentage of the model's answers that are informative.

- Metrics: Fine-tuned GPT-3 ("GPT-judge" / "GPT-info"), BLEURT, ROUGE, BLEU. The GPT-3 metrics are trained end-to-end to predict human evaluations of truthfulness and informativeness. BLEURT, ROUGE, and BLEU are used to compare the model's answer to each of the true and false reference answers. The score is then given by [max similarity to a true reference answer] - [max similarity to a false reference answer].

**Multiple-choice:**

While the generation task assesses a model's ability to say true statements, it is difficult to evaluate. We therefore provide a multiple-choice option that tests a model's ability to *identify* true statements.

- MC1 (Single-true): Given a question and 4-5 answer choices, select the only correct answer. The model's selection is the answer choice to which it assigns the highest log-probability of completion following the question, independent of the other answer choices. The score is the simple accuracy across all questions.

- MC2 (Multi-true): Given a question and multiple true / false reference answers, the score is the normalized total probability assigned to the set of true answers.



### **XSafety**

**📘Overview**

XSafety is the first multilingual safety benchmark for LLMs in response to the global deployment of LLMs in practice. 

Project Page: https://github.com/Jarviswang94/Multilingual_safety_benchmark

**📄 Benchmark Details**

XSafety covers 14 kinds of commonly used safety issues across 10 languages that span several language families. The authors utilize XSafety to empirically study multilingual safety for 4 widely-used LLMs, including both closed-API and open-source models. Experimental results show that all LLMs produce significantly more unsafe responses for non-English queries than English ones, indicating the necessity of developing safety alignment for non-English languages.



### **XSTest**

**📘Overview**

XSTest is a test suite to identify such eXaggerated Safety behaviours in a systematic way.

Paper：[XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://aclanthology.org/2024.naacl-long.301/)

Project Page: https://github.com/paul-rottger/xstest

**📄 Benchmark Details**

XSTest comprises 250 safe prompts across ten prompt types that well-calibrated models should not refuse to comply with, and 200 unsafe prompts as contrasts that models, for most applications, should refuse.









