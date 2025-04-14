## Prompt Engineering Techniques

**I. The Foundation: Understanding LLMs and Prompts**

- **Core LLM Mechanism:** At its heart, a Large Language Model (LLM) like Gemini is a sophisticated **prediction engine**. It processes input text sequentially, token by token (where a token is roughly a word or part of a word). Based on the vast amounts of text data it was trained on, it predicts the probability of every possible next token. It selects one, adds it to the sequence, and repeats the process. This iterative prediction is guided by the patterns, relationships, and structures learned during training.
- **The Prompt:** Your input to the LLM is the **prompt**. While primarily text, it can sometimes include other modalities like images if the model supports it (multimodal prompting, mentioned briefly as a separate concept). The prompt is the starting point, the seed from which the LLM generates its sequence of predicted tokens, forming the output.
- **Why Prompt Engineering Matters:** You don't need to be a data scientist to write _a_ prompt, but crafting an _effective_ prompt is crucial. A poorly constructed prompt can lead to:
    - **Ambiguity:** The model doesn't understand what you want.
    - **Inaccuracy:** The model generates factually incorrect information.
    - **Irrelevance:** The output doesn't address the actual need.
    - **Poor Formatting/Style:** The output isn't usable in the desired context. Effective prompt engineering guides the LLM's prediction process towards generating accurate, relevant, and useful output tailored to your specific task.
- **Iterative Nature:** Because so many factors influence the outcome (the specific model, its training, configuration settings, your wording, style, structure, context), prompt engineering is inherently an **iterative process**. You'll rarely get the perfect prompt on the first try. It involves crafting, testing, analyzing the output, refining the prompt (or configuration), and repeating until the desired result is achieved. This is why documentation becomes critical later on.
- **Context of Use (API/Vertex AI Focus):** While chatting with a bot involves prompting, the article specifically focuses on prompting models directly via APIs or platforms like Vertex AI. This distinction is important because direct access allows control over crucial **configuration parameters** (like Temperature, Top-K, Top-P, output length), which are often hidden or limited in simple chatbot interfaces. Mastering these configurations is a key part of advanced prompt engineering.

**II. Controlling the LLM's Output: Configuration is Key**

The prompt itself is only half the equation. The model's configuration settings significantly shape the output.

- **Output Length (Max Tokens):**
    
    - **What it does:** Sets a hard limit on the number of tokens the LLM will generate.
    - **Limitations:** It _doesn't_ make the LLM inherently more concise or stylistically brief. It simply _stops_the generation process once the limit is reached, potentially cutting off the response mid-thought or mid-sentence. If you need brevity, you must also engineer the _prompt_ to encourage it.
    - **Trade-offs:** More tokens mean more computation, leading to higher costs, potentially slower responses, and increased energy consumption.
    - **Relevance:** Essential for techniques like ReAct where the model might generate extraneous "thinking" steps after the answer; limiting output prevents excessive tokens.
- **Sampling Controls (Temperature, Top-K, Top-P): The Art of Randomness**
    
    - **Core Concept:** LLMs output _probabilities_ for the next token. Sampling controls determine how the model _chooses_ the single next token from this probability distribution.
    - **Temperature:** This is the primary dial controlling randomness.
        - `Low Temperature (e.g., 0 to 0.3)`: Makes the selection process more deterministic or "greedy." The model strongly favors the highest-probability tokens. `Temperature = 0` means it _always_picks the single most likely token (barring ties). Use this for tasks requiring factual accuracy, consistency, and predictability (e.g., extraction, classification, summarization of facts, math problems via CoT).
        - `High Temperature (e.g., 0.7 to 1.0+)`: Increases randomness, allowing less probable tokens a chance to be selected. This leads to more diverse, surprising, or "creative" outputs. Use this for brainstorming, creative writing, generating multiple options, or when using Self-Consistency. The softmax analogy helps: low temp sharpens the probability peak, high temp flattens it.
    - **Top-K:** Filters the candidate tokens _before_ sampling.
        - `Low K (e.g., 1 to 10)`: Considers only the top K most likely tokens. Highly restrictive. `K=1` is identical to Temperature 0 (greedy).
        - `High K (e.g., 40+)`: Considers a wider range of likely tokens, allowing more diversity than very low K but still filtering out highly improbable options. If K is set to the vocabulary size, it has no effect.
    - **Top-P (Nucleus Sampling):** Filters candidates based on cumulative probability.
        - It selects the _smallest set_ of most likely tokens whose probabilities sum up to at least P. For example, if P=0.9, it might select the top 3 tokens if their probabilities sum to 0.91, but it wouldn't consider the 4th token.
        - `Low P (e.g., 0.1)`: Very restrictive, often similar to greedy decoding (P=0 is effectively greedy).
        - `High P (e.g., 0.95 to 1.0)`: Considers a broader, dynamically sized set of tokens. `P=1.0`considers all tokens with non-zero probability. Top-P is adaptive; it might select very few tokens when the model is highly certain about the next word, and more tokens when it's uncertain.
    - **Interaction & Configuration Strategy:**
        - Typically, Top-K and/or Top-P act as filters first, then Temperature is applied to sample from the remaining candidates.
        - Extreme settings can override others (e.g., Temp 0 makes K/P irrelevant; K=1 makes Temp/P irrelevant).
        - Choosing between K and P (or using both) depends on the task and requires experimentation. The article provides starting points (e.g., Temp ~0.2, P~0.95, K~30 for coherent but slightly creative; higher values for more creativity; Temp 0 for factual tasks).
    - **The Repetition Loop Bug:** A common failure mode linked to sampling. Both very low temperatures (overly deterministic, gets stuck on the most likely path, which might loop) and very high temperatures (excessive randomness increases the chance of randomly returning to a previous state) can cause the model to repeat words or phrases endlessly. Careful tuning of sampling parameters is needed to avoid this.

**III. Core Prompting Techniques: Building Blocks for Interaction**

These are fundamental ways to structure your prompts.

- **Zero-Shot:** The simplest approach – just provide the instruction or question.
    - _Analysis:_ Easy to start with, works for simple tasks the model understands well from its training. However, it offers little control over the output format and may fail on complex reasoning or nuanced tasks. The movie classification example (Table 1) works because classification is a common task, but the ambiguity ("disturbing" vs. "masterpiece") highlights potential fragility. The importance of documenting even simple prompts (Table format) is introduced here.
- **Few-Shot (and One-Shot):** Providing one (one-shot) or multiple (few-shot) examples of the desired input-output pattern within the prompt itself.
    - _Analysis:_ This is arguably the single **most impactful** best practice highlighted. It directly _shows_ the model what you want, significantly improving its ability to follow instructions, match formatting (like the JSON pizza order example in Table 2), adopt a style, or handle specific task nuances. The quality of examples is paramount – they must be accurate, relevant, diverse, and cover edge cases if needed. A poor example can actively mislead the model. The number needed varies, requiring experimentation.
- **System, Contextual, and Role Prompting:** Techniques for adding layers of guidance.
    - _Analysis:_ These aren't mutually exclusive but offer different angles to shape behavior.
        - `System Prompts` set overarching rules or output requirements. They are powerful for enforcing structure (e.g., "Only return the label in uppercase" overriding high temperature in Table 3; generating valid JSON in Table 4), ensuring safety ("Be respectful"), or defining the model's core function for the interaction. The JSON output benefit (structure, less hallucination) is a key takeaway.
        - `Role Prompts` give the model a persona ("Act as a travel guide," Table 5). This effectively filters the model's knowledge and adjusts its tone, style, and vocabulary, making the output more relevant and engaging for specific scenarios. Adding stylistic qualifiers ("in a humorous style," Table 6) further refines this.
        - `Contextual Prompts` provide _immediate_, task-specific background information ("You are writing for a blog about retro 80's arcade video games," Table 7). This grounds the model in the current situation, preventing generic responses and ensuring relevance.

**IV. Advanced Prompting Techniques: Enhancing Reasoning and Interaction**

These techniques tackle more complex tasks requiring deeper reasoning or interaction with external knowledge.

- **Step-Back Prompting:** A clever two-step process: ask a related, more general question first, then use that answer as context for the specific question.
    - _Analysis:_ This forces the model to activate broader knowledge and principles before diving into specifics. As seen in the FPS storyline example (Tables 8-10), the direct prompt yielded a generic story, while the step-back approach (asking for engaging settings first) led to a much richer, more specific, and creative outcome based on one of those settings. It's a way to guide the model's internal "thought process" towards more relevant information spaces.
- **Chain of Thought (CoT):** Explicitly asking the model to "think step by step."
    - _Analysis:_ Addresses the LLM's weakness in multi-step reasoning (like math). By forcing it to articulate intermediate steps, the model is less likely to jump to incorrect conclusions. The age math problem (Tables 11-13) perfectly illustrates its power: failure without CoT, success with zero-shot CoT, and even adopting a specific reasoning _style_ with few-shot CoT. CoT provides interpretability (you see the reasoning) but increases output length/cost. The best practice of using Temp 0 for CoT makes sense, as logical reasoning often seeks a single correct path.
- **Self-Consistency:** An enhancement to CoT for dealing with ambiguity or tasks without a single "correct" reasoning path.
    - _Analysis:_ It leverages sampling (multiple CoT runs at high temperature) to explore diverse reasoning paths and then uses majority voting to find the most robust answer. The ambiguous email classification example (Table 14) shows how different runs can yield different conclusions; self-consistency provides a way to aggregate these into a more reliable result, albeit at a significantly higher computational cost.
- **Tree of Thoughts (ToT):** A further generalization, exploring multiple reasoning paths _in parallel_ like a tree search.
    - _Analysis:_ Conceptually powerful for complex exploration tasks, allowing the model to backtrack and pursue alternative lines of reasoning simultaneously, unlike the linear CoT. Less detail provided in the article, but represents an active area of research.
- **ReAct (Reason + Act):** Bridging the gap between the LLM's internal knowledge and the external world.
    - _Analysis:_ This is transformative. It allows the LLM to use _tools_ (like web search via SerpAPI in the Metallica kids example, Snippets 1-2). The LLM reasons about _what_ it needs to know, _acts_ by calling a tool, _observes_ the result, and incorporates it back into its reasoning. This overcomes the limitation of static training data and enables problem-solving requiring up-to-date information or interaction with other systems. It's complex to implement (managing state, API calls, parsing results) but unlocks significant capabilities, moving towards agent-like behavior.
- **Automatic Prompt Engineering (APE):** Using LLMs to help create prompts.
    - _Analysis:_ A meta-technique recognizing the difficulty of manual prompt crafting. An LLM generates candidate prompts/variations (like the T-shirt order examples, Table 15), which are then evaluated (often requiring another LLM or human judgment) to find the most effective ones. It automates part of the iterative discovery process.

**V. Specialized Application: Code Prompting**

LLMs are increasingly adept at handling code.

- **Writing Code:** Can generate code snippets from natural language descriptions (Bash script example, Table 16). _Crucial Caveat:_ LLMs mimic patterns, they don't truly understand semantics or guarantee correctness. **Thorough review and testing of generated code are non-negotiable.**
- **Explaining Code:** Can parse existing code and explain its functionality in natural language (Table 17). Useful for understanding unfamiliar codebases.
- **Translating Code:** Can convert code between different programming languages (Bash to Python example, Table 18). Useful for modernization or integration. The tip about using Markdown view in Vertex AI Studio for Python's indentation is a practical necessity.
- **Debugging & Reviewing Code:** Can analyze code and error messages to identify bugs and suggest fixes (Python `NameError` example, Table 19). Significantly, the example shows the LLM also proactively suggesting _further improvements_ beyond the immediate bug (extension handling, error checking), acting like a helpful code reviewer.

**VI. Essential Practices & Considerations**

These are the distilled wisdom points for effective prompt engineering:

- **Embrace Iteration & Experimentation:** This is the fundamental workflow.
- **Clarity & Simplicity:** Make prompts easy to understand.
- **Specificity:** Clearly define the desired output format, length, style, and content.
- **Instructions > Constraints:** Tell the model what to _do_ rather than just what _not_ to do.
- **Manage Length:** Use configuration or in-prompt requests.
- **Use Variables:** For dynamic and reusable prompts.
- **Experiment with Formats/Styles:** Try different ways of asking.
- **Mix Classes (Few-Shot Classification):** Avoid order bias.
- **Adapt to Models:** Stay updated and re-test.
- **Structured Output (JSON):** Leverage for clarity and consistency, but be aware of token cost and potential truncation.
- **JSON Repair & Schemas:** Use tools to fix broken JSON and schemas to structure input for clarity.
- **Collaborate:** Leverage team members' different approaches.
- **CoT Specifics:** Answer after reasoning, ensure extractability, use Temp 0.
- **DOCUMENT EVERYTHING:** Cannot be stressed enough. Use a template (like Table 21) to track prompts, configurations, outputs, goals, and results. This is vital for learning, reproducibility, debugging, and collaboration in the face of inherent LLM variability. Integrate documentation with saved prompts (e.g., in Vertex AI Studio) and eventually into version control and automated testing in production.

**VII. Synthesis**

Effective prompt engineering is far more than just typing a question. It's a systematic process that involves:

1. Understanding the underlying LLM prediction mechanism.
2. Precisely controlling output behavior through configuration settings (length, temperature, sampling).
3. Choosing the right prompting technique(s) for the task's complexity (from simple zero-shot to advanced reasoning chains like CoT or tool-using paradigms like ReAct).
4. Crafting clear, specific, and unambiguous instructions, often enhanced with well-chosen examples or contextual information.
5. Applying specialized techniques for domains like coding.
6. Adhering to best practices, particularly rigorous experimentation and meticulous documentation.

It's a blend of linguistic skill, logical thinking, technical configuration, and empirical science. 

---

## Prompt engineering for Large Language Models (LLMs)

**I. Foundational Principles of LLM Interaction via Prompts**

1. **LLM Operational Model:** LLMs function as autoregressive sequence predictors.1 They process input prompts via tokenization and predict subsequent tokens based on learned probability distributions derived from extensive training data. The core operation involves iteratively appending the predicted token and recalculating the distribution for the next token in the sequence.
2. **Prompt Definition:** A prompt constitutes the primary input mechanism (text, potentially augmented with other modalities like images per source [1]) used to elicit a desired output sequence from the LLM.
3. **Prompt Engineering Imperative:** While prompt creation is accessible, optimizing prompt efficacy is non-trivial. Efficacy is contingent upon multiple factors: the specific LLM employed (e.g., Gemini, GPT, Claude, Gemma, LLaMA per the text), its training dataset, configured hyperparameters, prompt lexicology, style, tone, structural composition, and contextual information provided.2 Suboptimal prompts yield ambiguous, inaccurate, or irrelevant outputs.3 Consequently, prompt engineering is defined as the iterative process of designing high-quality prompts to guide LLMs toward accurate and relevant outputs.4
4. **Context of Application:** The article distinguishes between general chatbot interaction and direct model interaction via APIs or platforms like Vertex AI. The latter provides essential access to configuration parameters (e.g., temperature, sampling controls), which is the primary focus of the advanced techniques discussed.

**II. Configuration Parameters and Output Control**

Direct interaction necessitates manipulation of model hyperparameters to modulate output characteristics:5

1. **Output Token Limit:** A constraint on the maximum number of tokens generated. This directly impacts computational cost, latency, energy consumption, and financial cost. It does not inherently induce stylistic conciseness but serves as a hard truncation point. This limit is particularly relevant for techniques like ReAct [13] that may generate intermediate verbose outputs.
2. **Sampling Strategies:** Methods governing the selection of the next token from the predicted probability distribution over the model's vocabulary.
    - **Temperature:** A parameter scaling the logits prior to softmax, effectively controlling the randomness of token selection.6 Low temperatures (approaching 0) approximate greedy decoding (selecting the maximum probability token, subject to tie-breaking variability), favoring determinism and coherence.7 High temperatures increase entropy, promoting diversity and creativity but risking incoherence. The analogy to the softmax temperature (T) parameter in machine learning is noted.
    - **Top-K Sampling:** Restricts the sampling pool to the K tokens with the highest probabilities. A low K enforces conservativeness (K=1 equals greedy decoding); a high K allows more diversity while excluding low-probability outliers.
    - **Top-P (Nucleus) Sampling:** Defined in [4], this technique selects the smallest set of tokens whose cumulative probability mass exceeds a threshold P. It dynamically adapts the size of the sampling pool based on the model's confidence (distribution sharpness). P=0 approximates greedy decoding; P=1 considers all tokens.
    - **Parameter Interaction:** When multiple sampling controls are active (e.g., in Vertex Studio), a typical implementation involves filtering candidates based on Top-K and/or Top-P criteria, followed by sampling from the filtered set using the Temperature setting. Extreme parameter values can render others ineffective (e.g., Temp=0 overrides K/P; K=1 overrides Temp/P). Recommended starting configurations are provided for varying creative requirements.
    - **Pathological Behavior (Repetition Loops):** A noted failure mode linked to sampling. Can occur at low temperatures (overly deterministic path fixation) or high temperatures (random chance leading back to previous states). Requires careful parameter tuning.

**III. Core Prompting Strategies**

Fundamental techniques for structuring prompts:

1. **Zero-Shot Prompting:** Providing only the task description without explicit examples.8 Suitable for simple, well-understood tasks but lacks fine-grained control. The importance of structured documentation (e.g., table format used in examples like Table 1) for tracking iterations is emphasized, referencing Vertex AI Studio [6].
2. **Few-Shot Prompting (incl. One-Shot):** Incorporating one or multiple input-output examples within the prompt (referenced in [7] for few-shot). This is highlighted as a highly effective best practice for demonstrating desired patterns, structures (e.g., JSON formatting in Table 2), or styles. Example quality (relevance, diversity, correctness, edge cases) and quantity (task-dependent, rule of thumb: 3-5) are critical factors, constrained by model input limits.
3. **System Prompting:** Defining overarching instructions, behavioral guidelines, or output format specifications (e.g., uppercase output in Table 3, JSON structure with schema in Table 4). Useful for enforcing constraints, ensuring safety ("Be respectful"), and obtaining structured data (reducing hallucinations, enabling downstream processing).
4. **Role Prompting:** Assigning a specific persona or character to the LLM (e.g., "Act as a travel guide," Table 5; adding "humorous style," Table 6). This influences tone, style, knowledge focus, and overall response framing. A list of potential styles is provided.
5. **Contextual Prompting:** Supplying immediate, task-specific background information to ground the LLM's response (e.g., specifying the blog's theme in Table 7).

**IV. Advanced Reasoning and Interaction Techniques**

Methods designed to elicit more complex reasoning or leverage external information:

1. **Step-Back Prompting:** Referenced in [6]. A two-stage technique: first prompting for general principles or related concepts, then using that output as context for the specific target prompt. Rationale: activates relevant knowledge domains, potentially mitigating biases and improving reasoning quality, as demonstrated by the FPS storyline example (Tables 8-10).
2. **Chain of Thought (CoT):** Referenced in [9]. Elicits explicit intermediate reasoning steps before the final answer (e.g., by adding "Let's think step by step."). Improves performance on tasks requiring multi-step logic (e.g., arithmetic, symbolic reasoning; Tables 11-13 illustrate failure without CoT and success with it).9 Can be combined with few-shot prompting. Advantages include low implementation effort, interpretability, and potentially improved robustness across model versions [10]. Disadvantage: increased token count (cost/latency). Best practices: place answer after reasoning, ensure answer extractability, use Temperature 0 for deterministic reasoning.
3. **Self-Consistency:** Referenced in [11]. Addresses limitations of greedy decoding in CoT by sampling multiple diverse reasoning paths (using high temperature) and selecting the final answer via majority vote. Improves accuracy and robustness, particularly for ambiguous tasks (e.g., email classification example, Table 14), at the cost of significantly increased computation.
4. **Tree of Thoughts (ToT):** Referenced in [12]. Generalizes CoT by allowing exploration of multiple reasoning paths concurrently in a tree structure (visualized in Figure 1). Theoretically advantageous for complex tasks requiring exploration and backtracking. Notebook reference [9 - likely typo for 12] mentioned.
5. **ReAct (Reason and Act):** Referenced in [13].10 A paradigm enabling LLMs to synergize reasoning with action execution via external tools (e.g., web search, code interpreter, APIs).11 Operates via a thought-action-observation loop, mimicking human problem-solving. Allows interaction with dynamic external information, representing a step towards agent modeling. Implementation requires managing state, tool integration (e.g., LangChain, SerpAPI example in Snippets 1-2), and parsing tool outputs. Notebook reference [14] provided.
6. **Automatic Prompt Engineering (APE):** Referenced in [15].12 A meta-technique using an LLM to generate candidate prompts for a task, which are subsequently evaluated (using metrics like BLEU/ROUGE or other methods) to identify optimal prompts. Aims to automate and potentially improve upon manual prompt design (e.g., T-shirt order variation generation, Table 15).13

**V. Domain-Specific Application: Code Prompting**

Leveraging LLMs for software development tasks:

1. **Code Generation:** Creating code snippets from natural language descriptions (Table 16). **Critical Warning:** Generated code requires rigorous human review and testing due to the LLM's lack of true semantic understanding.
2. **Code Explanation:** Generating natural language descriptions of code functionality (Table 17).
3. **Code Translation:** Converting code between programming languages (Table 18). Practical tip: use Markdown rendering in tools like Vertex AI Studio to preserve essential formatting (e.g., Python indentation).
4. **Code Debugging and Review:** Identifying errors from tracebacks, suggesting fixes, and providing broader code quality improvements (Table 19). Demonstrates capability beyond simple error identification.

**VI. Methodological Best Practices and Considerations**

Guidelines for systematic and effective prompt engineering:

1. **Iterative Process & Experimentation:** Emphasized throughout as fundamental. Requires testing variations in prompts, configurations, formats, and styles.14
2. **Clarity, Simplicity, Specificity:** Prompts should be unambiguous and clearly define the desired output characteristics. Use strong action verbs.
3. **Instructions vs. Constraints:** Prioritize positive instructions ("Do X") over negative constraints ("Do not do Y"), using constraints strategically for safety or strict formatting.
4. **Output Length Control:** Utilize configuration or in-prompt directives.
5. **Prompt Templating/Variables:** Employ variables (Table 20) for reusability and dynamic input, especially in application integration.
6. **Handling Classification Tasks (Few-Shot):** Mix the order of classes in examples to prevent overfitting to sequence and promote feature learning.
7. **Model Versioning:** Adapt and re-evaluate prompts when underlying LLM versions change.
8. **Structured Data Handling (JSON):**
    - **Output:** Leverage JSON for structured, consistent, and parseable output, despite higher token costs. Benefits include reduced hallucination and easier integration.
    - **Input:** Utilize JSON Schema (Snippet 5) to define expected input structure, guiding the model's attention and interpretation of complex input data (Snippet 6).
    - **Repair:** Employ tools like the `json-repair` library to handle potential JSON malformation due to token limit truncation.
9. **Collaboration:** Engage multiple individuals in prompt design efforts.
10. **Documentation:** Stressed as critically important due to inherent variability. A detailed template (Table 21) is recommended, tracking prompt versions, goals, models, configurations, full prompt text, outputs, results, and potentially links to saved prompts or RAG system parameters. Documentation should be integrated into development workflows, including version control and automated evaluation.

**VII. Summary Discrepancy**

The article's final summary list of techniques notably omits ReAct, Automatic Prompt Engineering, and the detailed Code Prompting section, despite dedicating significant discussion to them earlier. A comprehensive understanding should include these omitted areas.

This technical overview provides a structured synthesis of the concepts presented, emphasizing definitions, methodologies, cited sources, technical rationales, and practical considerations for advanced prompt engineering.