This overview delves into the world of prompt engineering for Large Language Models, or LLMs.

Let's begin with the basics: **What is a prompt?** When interacting with an LLM, a prompt is essentially your input. It's primarily text, though sometimes it might include other elements like images. This input is what the model uses to predict and generate a specific output. The good news? **You don't need to be a data scientist or a machine learning expert – absolutely everyone can write a prompt.**

However, crafting a _truly effective_ prompt can be more complex than it first appears. Why? Because many factors influence how well a prompt works. These include: the specific LLM you're using, the data that model was trained on, the model's configuration settings (which we'll discuss shortly), your choice of words, the style and tone you adopt, the structure of your prompt, and the surrounding context you provide. All of these elements matter.

This complexity means that **prompt engineering is often an iterative process.** You'll likely need to experiment and refine your prompts. If a prompt isn't well-crafted, it can lead to ambiguous or inaccurate responses, ultimately hindering the model's ability to give you the meaningful output you're looking for.

Now, while you technically write prompts when you chat with something like the Gemini chatbot – referenced as source [1] in the original text – this overview focuses more specifically on writing prompts when you interact with the underlying model _directly_, for example, through platforms like Vertex AI or using an API. The advantage here is gaining direct access to crucial model configurations, such as 'temperature', which significantly impact the output.

So, this discussion will explore prompt engineering in detail. We'll cover various techniques to get you started, share tips and best practices to help you become more adept, and also touch upon some common challenges you might encounter.

---

Let's dive deeper into **Prompt Engineering**.

First, it's crucial to remember the fundamental mechanism of an LLM: **it's a prediction engine.** It takes text as input, processes it sequentially, and then predicts what the very next 'token' (which is roughly a word or part of a word) should be. This prediction is based entirely on the patterns and relationships it learned from the massive dataset it was trained on. The LLM does this repeatedly – predicting a token, adding it to the sequence, and then predicting the _next_ token based on the updated sequence.

When you write a prompt, your goal is to structure that initial input in such a way that the LLM is guided to predict the _right sequence_ of tokens, ultimately forming the response you want. Therefore, **Prompt Engineering is the process of designing high-quality prompts that steer LLMs towards producing accurate and useful outputs.** This involves tinkering – experimenting to find the best wording, optimizing the length of your prompt, and evaluating how its writing style and structure align with the specific task you need the LLM to perform.

What kinds of tasks can effective prompts help with? A wide range! This includes text summarization, extracting specific pieces of information from larger texts, answering questions, classifying text into different categories, translating between languages or even programming code, generating new code, documenting existing code, or performing complex reasoning.

For practical examples and simple, effective prompting techniques, you can refer to Google's own prompting guides, cited as sources [2] and [3] in the text.

A key first step in prompt engineering is choosing your model. Whether you're using Gemini language models via Vertex AI, alternatives like GPT or Claude, or perhaps open-source models like Gemma or LLaMA, remember that **prompts often need to be optimized for the specific model** you've chosen to get the best results.

Beyond the prompt itself, you also need to get familiar with the model's configuration settings.

---

Let's talk about **LLM Output Configuration**.

After selecting your model, you'll need to consider its configuration options. Most LLMs provide several settings that control how the output is generated. Effective prompt engineering requires setting these configurations optimally for your specific task.

One important setting is the **Output Length**, which dictates the maximum number of tokens the model should generate in its response. It's important to know that generating more tokens requires more computation from the LLM. This leads to higher energy consumption, potentially slower response times, and consequently, higher costs.

However, simply reducing the output length limit doesn't automatically make the LLM write more concisely or succinctly. It just forces the model to _stop_ predicting tokens once that limit is hit, which might result in an abruptly cut-off response. So, if you need a short output, you'll likely need to engineer your _prompt_ to encourage brevity, in _addition_ to setting a lower token limit. Output length restrictions are especially critical for certain advanced prompting techniques, like ReAct, where a model might otherwise continue emitting useless tokens long after providing the actual answer you needed. Remember the trade-offs: more tokens mean more computation, energy, potentially time, and cost.

---

Next are the **Sampling Controls**.

LLMs don't just pick one single 'best' next token. Instead, they calculate probabilities for _all_ possible next tokens in their vocabulary. Sampling controls are the settings that determine how the model _chooses_ the single next token from this distribution of probabilities. The most common controls are Temperature, Top-K, and Top-P.

Let's look at **Temperature**. This setting controls the degree of randomness in the token selection process.

- **Lower temperatures** (closer to 0) make the selection more deterministic and focused. The model is highly likely to choose the token with the absolute highest probability. A temperature of exactly 0 is called 'greedy decoding' – it _always_ picks the single most likely token (though note, if multiple tokens share the exact same highest probability, tie-breaking mechanisms might introduce slight variations). Lower temperatures are generally good for prompts requiring factual, predictable answers.
- **Higher temperatures** increase randomness. Less likely tokens get a higher chance of being selected, leading to more diverse, creative, or sometimes unexpected results. As the temperature gets very high, all tokens become almost equally likely.
- The text provides an analogy: think of the Gemini temperature control like the softmax function in machine learning. A low temperature is like a low softmax 'T', strongly emphasizing the preferred choice. A high temperature is like a high softmax 'T', spreading the probability more broadly, making a wider range of options acceptable. This is useful for creative tasks where strict adherence to the single most probable path isn't desired.

---

Now for **Top-K and Top-P**.

These are two other common sampling settings used to control randomness by limiting the pool of tokens considered for the next step. They are referenced together under the term 'nucleus sampling' in source [4].

- **Top-K sampling** restricts the model's choice to only the top 'K' most likely tokens. For example, if K is 5, the model only considers the 5 tokens with the highest probabilities and chooses from that smaller set. A higher Top-K value allows for more varied and creative output because there are more options. A lower Top-K value makes the output more restricted and factual. A Top-K of 1 is equivalent to greedy decoding (always picking the single most probable token).
    
- **Top-P sampling**, also known as nucleus sampling, works slightly differently. It selects the smallest possible set of the most likely tokens whose cumulative probability meets or exceeds a certain threshold 'P'. For example, if P is 0.9 (or 90%), the model looks at the most likely token, then the next most likely, and so on, adding their probabilities together until the sum reaches 0.9. It then chooses the next token randomly _only_ from that selected set (the 'nucleus'). Values for P range from 0 (which is effectively greedy decoding) up to 1 (which considers all tokens in the vocabulary, weighted by their probabilities).
    

Which should you use, Top-K or Top-P? The best approach is often to **experiment** with both, or even use them in combination if the model allows, to see which configuration produces the results that best suit your specific needs.

---

Finally, **Putting It All Together**.

Choosing the optimal settings for output length, temperature, Top-K, and Top-P really depends on your specific application and the desired outcome. Importantly, these settings all **impact one another**. Adjusting one can change the effect of the others. It's also crucial to understand exactly how _your chosen model_implements and combines these different sampling strategies, as there can be variations.

Mastering both the art of crafting effective prompts and the science of tuning these configuration settings is key to unlocking the full potential of Large Language Models for your tasks.

---

Picking up where we left off, let's delve deeper into **Putting It All Together** regarding the LLM output configurations: Temperature, Top-K, and Top-P.

Choosing the right balance between these settings, along with the output token limit, is crucial. It depends heavily on your specific application and the kind of outcome you're aiming for. Remember, **these settings significantly impact one another.** It's also vital to understand how the specific model you're using actually combines these settings, as implementations can differ.

Let's consider a common scenario, like in Vertex Studio, where Temperature, Top-K, and Top-P are all available. Here's how they typically interact:

1. First, the model identifies tokens that meet _both_ the Top-K criterion (being among the K most probable tokens) _and_ the Top-P criterion (being part of the smallest set whose cumulative probability reaches P).
2. Then, the Temperature setting is applied to _sample_ from this filtered list of candidate tokens.

If only Top-K or only Top-P is available, the process is similar, just using the single available filter before sampling (if temperature is available) or randomly selecting (if temperature isn't available).

Now, consider what happens at **extreme settings**:

- If you set **Temperature to 0** (greedy decoding), then Top-K and Top-P become irrelevant. The single most probable token is always chosen. Conversely, if you set Temperature extremely high (e.g., above 1, perhaps into the 10s), its effect diminishes, and the choice becomes a near-random selection from the tokens that passed the Top-K and/or Top-P filters.
- If you set **Top-K to 1**, then Temperature and Top-P become irrelevant. Only one token qualifies, and it's always selected. If you set Top-K extremely high (like the size of the model's entire vocabulary), it effectively does nothing – almost all tokens pass the filter.
- If you set **Top-P to 0** (or very close to it), most implementations will only consider the single most probable token, making Temperature and Top-K irrelevant. If you set **Top-P to 1**, essentially all tokens with any probability pass the filter, making it ineffective.

As a **general starting point**, the article suggests:

- For relatively coherent results that allow some creativity: Try Temperature around **0.2**, Top-P around **0.95**, and Top-K around **30**.
- For _especially creative_ results: Start with Temperature around **0.9**, Top-P around **0.99**, and Top-K around **40**.
- For _less creative_, more factual results: Try Temperature around **0.1**, Top-P around **0.9**, and Top-K around **20**.
- If your task has a single, definitive correct answer (like a math problem): Start with **Temperature 0**.

**A note of caution:** Granting the LLM more freedom (higher temperature, K, P, and output tokens) increases the chance it might generate text that drifts off-topic or becomes less relevant.

**And a warning:** Have you ever encountered an LLM response that ends with repetitive filler words? This is sometimes called the **"repetition loop bug."** It happens when the model gets stuck generating the same word, phrase, or structure over and over. This can be exacerbated by poorly chosen Temperature and Top-K/P settings. Interestingly, it can occur at _both_ low and high temperatures:

- At _low_ temperatures, the model becomes overly deterministic, potentially getting stuck in a loop if the most probable path leads back to previously generated text.
- At _high_ temperatures, the excessive randomness increases the chance that the model might randomly pick a token that leads it back into a previous state, creating a loop by chance. In either case, the sampling gets stuck, leading to monotonous output. Solving this usually requires careful adjustment of Temperature and Top-K/P to find a better balance.

---

Now, let's shift gears to specific **Prompting Techniques**.

LLMs are designed to follow instructions and learn from vast amounts of data, enabling them to understand prompts and generate answers. However, they aren't perfect. **The clearer and more specific your prompt text, the better the LLM can predict the desired output.** Furthermore, certain techniques leverage how LLMs are trained and how they operate internally, helping you elicit more relevant and accurate results.

Let's explore some of the most important prompting techniques.

---

First up is **General Prompting**, also known as **Zero-Shot Prompting**.

This is the simplest form. You provide only a description of the task and perhaps some initial text for the LLM to work with. This could be a question, instructions, or the beginning of a story. It's called 'zero-shot' because you provide **zero examples** of the desired output format within the prompt itself.

The article shows an example using Vertex AI Studio (referenced as source [6]), a playground for testing prompts. The task is to classify a movie review as positive, neutral, or negative.

- **The Prompt:** Simply states "Classify movie reviews as POSITIVE, NEUTRAL or NEGATIVE." followed by the review text: _"Her" is a disturbing study revealing the direction humanity is headed if AI is allowed to keep evolving, unchecked. I wish there were more movies like this masterpiece.1 Sentiment:_
- **Configuration:** A low Temperature (0.1) is used because creativity isn't needed, just classification. Default Top-K and Top-P are used (which often effectively disable them). The token limit is set low (5) as only one word ("POSITIVE", "NEGATIVE", or "NEUTRAL") is expected.
- **Output:** The model correctly outputs "POSITIVE", despite the potentially confusing use of "disturbing" alongside "masterpiece".

The example uses a table format to document the prompt (Name, Goal, Model, Config, Prompt, Output). The article stresses that this is a **great practice**. Your prompts will likely evolve, so keeping structured track of your experiments is crucial for effective prompt engineering. More on documentation is promised later in the Best Practices section.

When zero-shot prompting doesn't quite work, or you need more control over the output format, you can provide examples...

---

...which brings us to **One-Shot & Few-Shot Prompting**.

Providing examples within your prompt is a powerful technique. These examples help the model better understand _exactly_ what you're asking for, especially when you need the output to follow a specific structure or pattern.

- **One-Shot Prompting:** Provides a _single_ example (hence 'one-shot'). The model uses this one demonstration as a template to imitate for the actual task.
- **Few-Shot Prompting:** Provides _multiple_ examples (referenced in source [7]). This reinforces the desired pattern, increasing the likelihood the model will follow it correctly.

How many examples do you need for few-shot? It depends on the task complexity, the quality of your examples, and the model's capabilities. A general rule of thumb is **at least three to five examples**. However, complex tasks might need more, while model input length limitations might force you to use fewer.

The article provides a few-shot example (Table 2): Parsing pizza orders into JSON format.

- **Configuration:** Same low temperature (0.1) as before, but a higher token limit (250) to accommodate the JSON output.
- **Prompt:** It starts with the instruction "Parse a customer's pizza order into valid JSON:". Then it provides two distinct `EXAMPLE:` blocks, each showing a natural language order ("I want a small pizza...") followed by the correctly formatted `JSON Response:` structure. After the examples, it presents the _new_order to be parsed ("Now, I would like a large pizza, half cheese...").
- **Output:** The model successfully generates the correct JSON for the new, more complex half-and-half order, following the pattern established by the examples.

**Crucially, the quality of your examples matters immensely.** They should be:

- **Relevant:** Directly related to the task.
- **Diverse:** Covering different variations.
- **High Quality:** Accurate and well-written. A single mistake in an example can confuse the model.
- **Include Edge Cases:** If you need robust handling of unusual inputs, make sure to include examples of those edge cases.

---

Next, let's discuss **System, Contextual, and Role Prompting**.

These are three techniques used to guide LLM text generation, each focusing on a slightly different aspect:

- **System Prompting:** Sets the _overall context and purpose_ for the model. It defines the 'big picture' – what fundamental task the model should be performing (e.g., act as a translator, classify text).
- **Contextual Prompting:** Provides _specific details or background information_ relevant to the _current_ task or conversation. It helps the model understand the nuances of the immediate request. This context is often dynamic.
- **Role Prompting:** Assigns a specific _character, persona, or identity_ for the model to adopt. This influences the tone, style, and knowledge base the model uses in its response.

While there can be considerable overlap (e.g., a prompt might assign a role _and_ provide context), distinguishing them helps clarify intent:

- System: Defines _what_ the model fundamentally is or does for the task.
- Contextual: Provides _immediate, task-specific information_.
- Role: Frames the _output style, voice, and persona_.

This framework aids in designing prompts deliberately and analyzing how each part influences the final output. Let's look at each individually.

---

**System Prompting** involves giving the model additional instructions on _how_ it should behave or format its output, beyond just the core task.

Table 3 shows an example returning to the movie classification task:

- **Prompt:** "Classify movie reviews as positive, neutral or negative. **Only return the label in uppercase.**" followed by the review.
- **Configuration:** Interestingly, Temperature is set high (1), Top-K is 40, Top-P is 0.8, normally encouraging creativity. Token limit is 5.
- **Output:** "NEGATIVE". Despite the high creativity settings, the _clear system instruction_ ("Only return the label in uppercase") overrides the tendency for verbose output, resulting in just the single required word.

System prompts are useful for enforcing specific output requirements. The name implies providing an _additional task_ to the 'system'. For example, specifying a programming language for code generation, or, as shown in Table 4, requesting a specific **JSON structure**.

In Table 4:

- **Prompt:** Asks for classification but adds "Return valid JSON:" and provides a `Schema:` defining the desired JSON structure (`{"movie_reviews": [{"sentiment": "...", "name": "..."}]}`).
- **Output:** The model generates the review information precisely in the requested JSON format.

**Benefits of using JSON output** via system prompts include:

- Automation: Easier parsing in downstream applications.
- Sorting: Can instruct the model to return data sorted (e.g., by date).
- Structure & Reduced Hallucination: Forcing the model into a defined structure can limit irrelevant or made-up information.

System prompts are also valuable for **safety and toxicity control**. Adding a simple instruction like "You should be respectful in your answer" can guide the model's behavior.

---

**Role Prompting** involves assigning a specific role or persona to the model.

This helps the model generate more relevant and informative output by tailoring its responses, tone, and knowledge to fit the assigned identity. You could ask it to be a book editor, a kindergarten teacher, a technical expert, or, as in Table 5, a travel guide.

Table 5 Example:

- **Prompt:** "I want you to act as a travel guide. I will write to you about my location and you will suggest 3 places to visit near me... My suggestion: 'I am in Amsterdam and I want to visit only museums.'"
- **Configuration:** High creativity settings (Temp 1, Top-K 40, Top-P 0.8), large token limit (1024).
- **Output:** Provides three relevant museum suggestions in Amsterdam (Rijksmuseum, Van Gogh Museum, Stedelijk Museum) with appropriate descriptions, consistent with a travel guide role.

The article notes that changing the role (e.g., to a geography teacher) would result in a different kind of response, even with the same location input. Assigning a role provides a blueprint for the desired **tone, style, and focused expertise**, improving output quality, relevance, and effectiveness.

You can even specify _styles_. The article lists effective styles like Confrontational, Descriptive, Formal, Humorous, Inspirational, Persuasive, etc.

Table 6 demonstrates this by modifying the travel guide prompt:

- **Prompt:** "I want you to act as a travel guide... suggest 3 places to visit near me in **a humorous style.**My suggestion: 'I am in Manhattan.'"
- **Output:** Suggests the Empire State Building, MoMA, and Fifth Avenue, but describes them with humorous flair ("feel like King Kong," "make you question whether your stick-figure drawings have any artistic merit," "make your wallet cry").

---

**Contextual Prompting** focuses on providing relevant background information _for the specific task at hand_.

This helps the model quickly grasp the nuances of your request, leading to more seamless interactions and more accurate, relevant responses.

Table 7 provides an example:

- **Goal:** Suggest blog articles about retro games.
- **Prompt:** Starts with `Context: You are writing for a blog about retro 80's arcade video games.` Then asks: `Suggest 3 topics to write an article about...`
- **Configuration:** High creativity settings, large token limit.
- **Output:** Suggests relevant article topics ("Evolution of Arcade Cabinet Design," "Iconic Arcade Games of The 80's," "Rise and Retro Revival of Pixel Art") with descriptions appropriate for that specific blog context.

---

Finally, let's look at **Step-Back Prompting** (referenced with source [6], likely the same source as the Vertex AI Studio reference earlier).

This technique aims to improve performance on complex tasks by prompting the LLM in two stages:

1. **Step Back:** First, ask the LLM a _general question_ related to the underlying principles or concepts of the specific task.
2. **Specific Task:** Then, feed the LLM's answer to that general question _as context_ into a second prompt that asks for the original, specific task.

The idea is that this 'step back' activates relevant background knowledge and reasoning processes within the LLM _before_ it tackles the specific problem. By considering broader principles, the model can generate more accurate, insightful, and potentially less biased responses. It essentially encourages the model to "think" more deeply about the topic by utilizing more of its internal knowledge.

The article illustrates this with writing a storyline for a first-person shooter (FPS) game level:

- **Table 8 (Traditional Prompt):** Asks directly: "Write a one paragraph storyline for a new level of a first-person shooter... that is challenging and engaging." Configuration is high creativity (Temp 1). The output is okay, but quite generic (ambush in urban area, fight through alleys, uncover intel).
- **Table 9 (Step-Back - General Question):** Asks: "Based on popular first-person shooter action games, what are 5 fictional key settings that contribute to a challenging and engaging level storyline...?" The model outputs 5 distinct, evocative settings (Abandoned Military Base, Cyberpunk City, Alien Spaceship, Zombie Town, Underwater Facility).
- **Table 10 (Specific Task with Context):** Takes the 5 settings generated in the step-back phase and provides them as `Context:`. Then, it asks the original question again: "Take one of the themes and write a one paragraph storyline...".
- **Result:** The output selects the "Underwater Research Facility" theme and generates a much more specific, detailed, and arguably more interesting storyline based on that concrete setting.

By using this step-back technique, prompting for the general principles first and then applying them to the specific task, you can often significantly increase the quality and accuracy of the LLM's output for complex reasoning or creative tasks.

---

Let's continue exploring powerful prompting techniques, starting with **Chain of Thought (CoT)**, referenced as source [9].

Chain of Thought prompting is designed to enhance the reasoning abilities of Large Language Models. The core idea is to instruct the model not just to give an answer, but to **generate intermediate reasoning steps**– essentially, to 'show its work' like a human solving a problem. This step-by-step process helps the LLM arrive at more accurate conclusions, especially for tasks requiring logical deduction or calculation.

You can often combine CoT with the few-shot prompting technique we discussed earlier. Providing examples that _include_ reasoning steps is particularly effective for complex tasks where a simple zero-shot "think step by step" instruction might not be enough.

CoT offers several advantages:

- It's relatively **low-effort** to implement but can be very effective.
- It works well with **off-the-shelf LLMs**, meaning you often don't need to fine-tune the model specifically.
- It provides **interpretability**. You can see the LLM's reasoning process, understand how it reached the answer, and potentially identify where it went wrong if there's an error.
- It appears to improve **robustness** when migrating between different LLM versions or models, meaning your prompt's performance might degrade less over time compared to prompts without explicit reasoning chains.

The main disadvantage is intuitive: generating the reasoning steps means **more output tokens**. This increases the cost and the time it takes to get a response.

Let's see CoT in action with an example math problem (Table 11). The prompt is: "When I was 3 years old, my partner was 3 times my age. Now, I am 20 years old. How old is my partner?" Without CoT, the model incorrectly outputs "63 years old." This highlights a common issue: LLMs, trained primarily on text, often struggle with mathematical reasoning.

Now, let's apply Zero-Shot CoT (Table 12). We add a simple phrase to the prompt: "... How old is my partner? Let's think step by step."

The output now shows the reasoning:

1. Identifies current age variable 'x'.
2. States age was 3.
3. Calculates partner's age then (3 * 3 = 9).
4. Calculates the time elapsed (20 - 3 = 17 years).
5. Applies elapsed time to partner's age (9 + 17 = 26).
6. States the final answer: 26 years old. This time, the answer is correct because the model was forced to break down the problem. The article notes the model's logic (adding elapsed time) differs slightly from a common human approach (calculating the fixed age difference: 9-3=6, then adding it to the current age: 20+6=26).

This leads to **Few-Shot CoT** (Table 13). Here, we provide _one complete example_ (Q&A including the step-by-step reasoning using the age difference method) before posing the target question.

- **Prompt:** Includes a Q&A pair about a brother's age, demonstrating the reasoning `(age difference = 4-2=2; brother's age = 40-2=38)`. Then it asks the original partner age question, again ending with "Let's think step by step. A:"
- **Output:** The model now follows the _style_ of reasoning shown in the example: "When I was 3 years old, my partner was 3 * 3 = 9 years old. That's an age difference of 6 years... Now I am 20 years old, so my partner is 20 + 6 = 26 years old. The answer is 26."

CoT is useful for various use cases beyond math:

- **Code generation:** Breaking down requirements into steps before writing code.
- **Synthetic data creation:** Guiding the model through assumptions (e.g., "Write a product description for XYZ, assume it's targeted at...")
- Essentially, any task that benefits from 'talking through' the solution process.

For more details on CoT, the article points to a notebook [10] hosted in the GoogleCloudPlatform Github repository. Best practices specific to CoT are mentioned to be covered later.

---

Building on CoT, we have **Self-Consistency**, referenced as source [11].

While CoT improves reasoning, standard implementations often use 'greedy decoding' (just picking the single most likely next step), which can still lead down an incorrect path. Self-consistency aims to improve accuracy and coherence further by **sampling multiple diverse reasoning paths and selecting the most frequent answer.**

It works like this:

1. **Generate Diverse Paths:** Prompt the LLM with the _same_ CoT prompt multiple times, but use a _high temperature_ setting. This encourages the model to explore different ways of thinking and generate varied reasoning chains.
2. **Extract Answers:** Pull out the final answer from each generated response.
3. **Majority Vote:** Choose the answer that appears most frequently across all the generated paths.

This gives a sort of pseudo-probability for an answer being correct. The obvious downside is **high cost**, as you're running the same prompt multiple times.

Let's look at the email classification example (Table 14). The task is to classify an email as IMPORTANT or NOT IMPORTANT. The email itself is tricky: it reports a potential security bug (JavaScript injection via contact form name field) but uses a friendly, sarcastic tone ("Feel free to leave the bug... it gives me more interesting things to read") and signs off as "Harry the Hacker." This ambiguity could easily confuse an LLM.

The prompt asks the LLM to classify the email and "Let's think step by step and explain why." The article shows the results of running this prompt three times (with high temperature implied):

- **Attempt 1:** Reasons about the potential impact of the bug and sender credibility, concludes **IMPORTANT**.
- **Attempt 2:** Focuses on lack of urgency, non-critical description by sender, lack of personal impact, and concludes **NOT IMPORTANT**.
- **Attempt 3:** Similar reasoning to Attempt 1, focusing on the security risk and unknown sender credibility, concludes **IMPORTANT**.

By generating these multiple chains of thought, we see the model's uncertainty. Applying self-consistency, we take the majority vote: "IMPORTANT" appeared 2 out of 3 times. This provides a more robust answer than relying on a single, potentially flawed, reasoning path.

---

Next is **Tree of Thoughts (ToT)**, referenced as source [12].

ToT generalizes Chain of Thought. Instead of following a single, linear sequence of reasoning steps, ToT allows the LLM to **explore multiple different reasoning paths _simultaneously_**, creating a tree-like structure. Figure 1 in the article visualizes this contrast: CoT is a line, ToT is a branching tree.

This parallel exploration makes ToT potentially better suited for complex problems that require significant exploration or backtracking. The model maintains a 'tree' where each node is a 'thought' (an intermediate reasoning step). It can then branch out from different nodes to explore alternative pathways.

For more details, the article mentions a notebook based on the ToT paper, referenced again as source [9] (likely a typo in the original article, should be [12]).

---

Now for **ReAct (Reason and Act)**, referenced as source [13] (originally [10] in the text, adjusted based on user's list).

ReAct is a powerful paradigm that combines **natural language reasoning** with the ability for the LLM to **use external tools**. This allows the LLM to perform actions like searching the web, running code, or interacting with APIs to gather information it doesn't inherently possess. This is a crucial step towards creating autonomous AI agents.

ReAct mimics how humans solve problems: we reason verbally, but we also take actions (like looking something up) to get more information. It works through a **thought-action loop**:

1. **Thought:** The LLM reasons about the problem and devises a plan, which might include needing external information.
2. **Action:** The LLM decides to use a specific tool (e.g., Search) with a specific input.
3. **Observation:** The LLM receives the result back from the tool.
4. **Thought:** The LLM incorporates the observation into its reasoning and updates its plan or determines the next step.
5. This loop continues until the LLM reaches the final answer.

The article demonstrates this with code using the LangChain framework, VertexAI, and a search tool (SerpAPI). **Note:** Running this requires setting up an API key for the search tool (SerpAPI in this case) as an environment variable (`SERPAPI_API_KEY`).

The task given to the ReAct agent (Code Snippet 1 shows setup) is: "How many kids do the band members of Metallica have?"

Code Snippet 2 shows the execution trace:

- The agent starts (`Entering new AgentExecutor chain...`).
- **Thought:** Metallica has 4 members. Need to find kids for each.
- **Action:** Search [How many kids does James Hetfield have?]
- **Observation:** three children
- **Thought:** 1/4 members found, total 3 kids. Need Lars Ulrich next.
- **Action:** Search [How many kids does Lars Ulrich have?]
- **Observation:** 3
- **Thought:** 2/4 members found, total 3 + 3 = 6 kids. Need Kirk Hammett next.
- **Action:** Search [How many kids does Kirk Hammett have?]
- **Observation:** ...two sons...
- **Thought:** 3/4 members found, total 6 + 2 = 8 kids. Need Robert Trujillo next.
- **Action:** Search [How many kids does Robert Trujillo have?]
- **Observation:** 2
- **Thought:** 4/4 members found, total 8 + 2 = 10 kids. Found all, can give final answer.
- **Final Answer:** 10

This clearly shows the LLM reasoning, deciding to act (search), observing the result, and updating its internal state until the task is complete.

Practical use of ReAct involves managing the conversation history (feeding back previous thoughts/actions/observations), potentially trimming unnecessary text generated by the LLM during its 'thoughts', and setting up the agent with appropriate instructions or examples. For a more detailed look at the LLM inputs/outputs in a ReAct process, refer to the notebook mentioned as source [14].

---

Given the complexity of crafting prompts, wouldn't it be nice to automate it? That's the idea behind **Automatic Prompt Engineering (APE)**, referenced as source [15].

APE uses an LLM to **help generate and refine prompts** for a specific task. It aims to reduce manual effort and potentially discover prompts that perform better than human-designed ones.

The general process is:

1. **Generate Candidates:** Prompt an LLM to generate multiple candidate prompts or instruction variations for your target task.
2. **Evaluate:** Score these generated candidates using a relevant metric (like BLEU or ROUGE for text similarity, or task-specific accuracy, or even human rating).
3. **Select:** Choose the highest-scoring prompt. You might also manually tweak the best candidates and repeat the evaluation.

The article provides an example for training a chatbot for a band merchandise webshop. The goal is to get variations of the customer order: "One Metallica t-shirt size S".

- **Step 1 (Table 15):** Prompt gemini-pro: "Generate 10 variants, with the same semantics but keep the same meaning." The output includes variations like "I'd like to purchase a Metallica t-shirt in size small," "Can I order a small-sized Metallica t-shirt?," "One Metallica shirt, size small, please," etc.
- **Step 2:** Evaluate these 10 variants using a chosen metric.
- **Step 3:** Select the best-performing variant(s) to use for chatbot training or in the application logic.

---

Finally, let's touch upon **Code Prompting**.

Gemini models excel at text, and this includes generating, explaining, and translating programming code. Vertex AI Studio is recommended for this, especially for confidentiality and access to configuration settings like temperature.

1. Prompts for Writing Code:

LLMs can act as coding assistants, speeding up development. The example involves creating a Bash script to rename files in a folder by prepending "draft_".

- **Prompt (Table 16):** Clearly describes the desired functionality: ask for folder name, check existence, loop through files, rename using `mv` command with the "draft_" prefix. Uses low temperature (0.1) for factual code generation.
- **Output:** Generates a documented Bash script.
- **Crucial Warning Reiterated:** LLMs don't _understand_ code deeply; they replicate patterns from training data. **Always thoroughly review, test, and verify any code generated by an LLM before using it.**
- **Verification:** The article confirms the generated script was saved, executed, and worked correctly on a test folder.

2. Prompts for Explaining Code:

LLMs can help decipher code written by others.

- **Prompt (Table 17):** Takes the previously generated Bash script (with comments removed) and asks "Explain to me the below Bash code:".
- **Output:** Provides a step-by-step explanation of the script's logic (user input, folder check, file listing, renaming loop, success message).

3. Prompts for Translating Code:

LLMs can convert code between programming languages. The example translates the Bash renaming script to Python, aiming for better reusability, perhaps for a web application.

- **Prompt (Table 18):** Provides the Bash code and asks "Translate the below Bash code to a Python snippet."
- **Output:** Generates equivalent Python code using the `os` and `shutil` modules.
- **Testing:** Mentions saving the Python code and testing it.
- **Vertex AI Studio Tip:** When generating code (especially Python) in Vertex AI Studio, make sure to view the output in **Markdown** format. This preserves crucial indentation that is syntactically required for Python to run correctly. Viewing as plain text might lose this formatting.

This concludes our detailed overview of the prompting techniques and considerations discussed in the article.

---

Continuing our exploration of prompt engineering, let's revisit **Code Prompting**, specifically focusing on **Debugging and Reviewing Code**.

Imagine we've edited the Python script from Table 18 (which renamed files) to make it more flexible. We want it to prompt the user for the filename prefix and ensure that prefix is in uppercase. Snippet 3 shows this edited code. We've introduced a line `text = toUpperCase(prefix)`... but uh oh, running this code (Snippet 4) results in a Python error: `NameError: name 'toUpperCase' is not defined`. We broke it!

Can an LLM help us fix this? Yes. Table 19 demonstrates prompting for debugging.

- **The Prompt:** We provide the LLM with the exact error message (the Traceback ending in `NameError`) and the broken Python code (Snippet 3). We explicitly ask it to "Debug what's wrong and explain how I can improve the code." We use a low temperature (0.1) for a factual response.
- **The Output:** The LLM does several helpful things:
    1. It correctly identifies the `NameError`, explaining that `toUpperCase` is not a standard Python function.
    2. It provides the correct fix: use the built-in string method `.upper()`, changing the line to `text = prefix.upper()`.
    3. It includes the corrected code snippet.
    4. **Impressively**, it goes beyond the immediate bug fix and reviews the code for further improvements, suggesting:
        - Preserving the original file extension during renaming (using `os.path.splitext`).
        - Considering how to handle spaces in folder names (though it doesn't implement this).
        - Acknowledging the good use of f-strings.
        - Adding error handling using a `try...except` block around the `shutil.move` call to catch potential issues during renaming.
    5. It even provides a _second_, improved code snippet incorporating the extension handling and error catching logic. (The article notes this response was truncated due to token limits, highlighting the need to manage output length).

This example powerfully illustrates that LLMs can be valuable debugging partners, not only pinpointing specific errors but also offering insightful suggestions for improving code robustness and readability.

---

Before diving into best practices, the article briefly touches on **What about multimodal prompting?**

It clarifies that multimodal prompting – using inputs beyond just text, such as images, audio, or combinations thereof – is a separate concept from the text and code prompting discussed extensively so far. It depends entirely on the specific capabilities of the LLM being used.

---

Now, let's consolidate the wisdom shared throughout the article into a set of **Best Practices** for effective prompt engineering. Remember, finding the perfect prompt often requires tinkering, and tools like Language Studio in Vertex AI provide an excellent playground for this experimentation.

1. **Provide Examples (Few-Shot):** This is highlighted as perhaps the _most important_ best practice. Giving the model one or more examples (one-shot or few-shot) of the desired output acts as a powerful teaching mechanism. It shows the model the target format, style, and content, significantly improving the accuracy and relevance of its generated response.
    
2. **Design with Simplicity:** Aim for prompts that are concise, clear, and easy to understand – both for you and the model. If a prompt feels convoluted to you, it likely will be for the LLM too. Avoid jargon and unnecessary information. Use strong, direct action verbs. The article contrasts a vague request ("I am visiting New York... where should we go?") with a clear, role-based instruction ("Act as a travel guide... Describe great places to visit in New York Manhattan with a 3 year old."). It also lists useful action verbs like Analyze, Classify, Create, Describe, Extract, Generate, Summarize, Translate, etc.
    
3. **Be Specific About the Output:** Don't be vague. Instead of "Generate a blog post about video game consoles," specify details like length ("3 paragraph"), content focus ("top 5 consoles"), and style ("informative and engaging," "conversational style"). Use system or contextual prompting to provide these specifics.
    
4. **Use Instructions over Constraints:** Guide the model by telling it _what to do_ (Instructions) rather than just _what not to do_ (Constraints). Research suggests positive instructions are often more effective, clearer, and allow for more flexibility. While constraints have their place (e.g., for safety, preventing bias, enforcing strict formats), prioritize instructions first. Example: Instead of "Do not list video game names," say "Only discuss the console, company, year, and sales." Use constraints judiciously when needed.
    
5. **Control the Max Token Length:** Manage response length either through the model's configuration settings or by making explicit requests within the prompt, like "Explain quantum physics in a tweet length message."
    
6. **Use Variables in Prompts:** For reusability and dynamic prompts, especially when integrating with applications, use variables (like placeholders). Table 20 shows an example using `{city}` as a variable in a travel guide prompt, allowing the same prompt structure to be used for different cities.
    
7. **Experiment with Input Formats and Writing Styles:** Recognize that results vary significantly based on the model, configuration, prompt wording, structure, and even subtle phrasing. Don't be afraid to experiment. The article shows how phrasing a request about the Sega Dreamcast as a question, a statement, or an instruction can lead to different outputs.
    
8. **For Few-Shot Classification, Mix Up Classes:** While example order might not always matter, for classification tasks, ensure your few-shot examples represent different classes in a mixed order. This prevents the model from simply learning the sequence and encourages it to identify the actual features of each class, leading to more robust performance. A starting point of around 6 examples is suggested.
    
9. **Adapt to Model Updates:** LLMs evolve. Stay informed about updates to model architecture, training data, and capabilities. Re-test your existing prompts with newer model versions and adapt them to leverage new features. Tools like Vertex AI Studio help manage and test prompt versions.
    
10. **Experiment with Output Formats (Structured Data):** Especially for non-creative tasks like data extraction, parsing, or classification, consider instructing the model to return output in structured formats like JSON or XML. The benefits, reiterated here, include consistency, focus on desired data, reduced hallucination, explicit relationships, data typing, and potential for sorting. Table 4 is referenced as an example.
    
11. **JSON Repair:** A practical tip when working with JSON output: LLMs might produce truncated or malformed JSON, especially if hitting token limits. Tools like the `json-repair` Python library can automatically attempt to fix these issues, making structured output more reliable.
    
12. **Working with Schemas (for Input):** Just as JSON is useful for output, JSON Schema can structure your _input_. Providing a schema defines the expected data structure and types, helping the LLM focus on relevant information and understand relationships, especially with complex or large datasets. Snippets 5 and 6 show defining a product schema (name, category, price, features, release_date) and then providing product data conforming to that schema, enabling the LLM to generate a better description.
    
13. **Experiment Together:** If working in a team, encourage multiple people to attempt prompt engineering for the same task. Different approaches, while following best practices, can yield varied results and lead to discovering highly effective prompts faster.
    

---

The article then provides some **CoT Best practices** specifically for Chain of Thought prompting:

- **Answer Placement:** Always place the final answer _after_ the reasoning steps in your prompt examples or instructions. The process of generating the reasoning affects the model's internal state, influencing the final answer prediction.
- **Answer Extraction:** Design your CoT prompts so that the final answer is clearly distinguishable and separable from the preceding reasoning chain. This is crucial for evaluation and especially when using Self-Consistency.
- **Temperature:** For standard CoT aimed at logical reasoning or problems with a single correct answer, set the **Temperature to 0**. This encourages greedy decoding, following the most probable path, which aligns with finding a deterministic solution.

---

Finally, the article reiterates the critical importance of **Documenting the Various Prompt Attempts**.

Given the variability in LLM outputs (across models, settings, versions, and even run-to-run randomness), meticulous documentation is essential. It allows you to track progress, revisit past work, compare performance across model updates, and debug issues effectively.

A template (like Table 21, possibly implemented in a Google Sheet) is recommended, capturing:

- Prompt Name and Version
- Goal
- Model Name and Version
- Configuration Settings (Temperature, Token Limit, Top-K, Top-P)
- The Full Prompt Text
- The Output(s)

Additionally, tracking the prompt iteration number, result status (OK/NOT OK/SOMETIMES), feedback comments, and hyperlinks to saved prompts (e.g., in Vertex AI Studio) is highly beneficial. For RAG systems, documenting the retrieval specifics (query, chunk settings, etc.) is also important.

Once a prompt is deemed effective, integrate it into your application, preferably storing prompts separately from code for easier maintenance. Rely on automated testing and evaluation in production environments.

Remember: **Prompt engineering is iterative.** Craft, test, analyze, document, refine, and repeat, especially when models or requirements change.

---

**Summary**

To wrap up, this whitepaper explored the field of prompt engineering, covering fundamental concepts and diving into various techniques including: Zero-shot, Few-shot, System, Role, and Contextual prompting, as well as more advanced methods like Step-back, Chain of Thought, Self-Consistency, and Tree of Thoughts. It also touched upon ReAct, Automatic Prompt Engineering, and specific strategies for Code Prompting, concluding with a comprehensive set of best practices to guide your prompt engineering efforts.

---
---

Here is an analysis of the information derived from the provided reference links, contextualizing their relevance to the original article's discussion on prompt engineering.

**1. Google Gemini Product ([Ref 1: gemini.google.com])**

- **Source Information:** This refers to the user-facing interface and potentially the underlying family of Google's Gemini AI models. Features highlighted include conversational interaction (text, voice, image), capabilities for writing assistance, brainstorming, learning, summarization (integrating with Gmail/Drive), image generation, and mobile assistant functionalities (on Android).
- **Relevance to Article:** Establishes the primary LLM family [1] discussed and used in examples (e.g., `gemini-pro` model). It provides context for the types of tasks (summarization, generation, Q&A) that prompt engineering aims to optimize for these models.

**2. Gemini for Google Workspace Prompt Guide ([Ref 2: Google Workspace Guide])**

- **Source Information:** This guide offers practical, user-centric advice for prompting Gemini within the Workspace suite (Docs, Gmail, Sheets, etc.). Key recommendations include using natural language, clarity, conciseness, providing context, using relevant keywords, and decomposing complex tasks into simpler prompts. It emphasizes that clear prompts yield better, more reliable results and improve efficiency.
- **Relevance to Article:** While the main article focuses on more technical API/Vertex AI prompting, this guide [2] reinforces the universal best practices of clarity, context, and specificity mentioned later in the article's "Best Practices" section. It serves as a foundational reference for basic prompt construction principles.

**3. Introduction to Prompting (Google Cloud Vertex AI) ([Ref 3: Vertex AI Docs])**

- **Source Information:** This documentation provides a formal introduction to prompt design specifically for models within Google Cloud's Vertex AI platform. It confirms that prompts for models like Gemini can contain instructions, context, few-shot examples, etc., and explicitly directs users to Vertex AI Studio for experimentation.
- **Relevance to Article:** Directly cited [3] as a foundational resource. It confirms the article's technical focus on prompting within the Vertex AI ecosystem and validates the core components of a prompt discussed.

**4. Text Model Request Body: Sampling Methods (Google Cloud Vertex AI) ([Ref 4: Vertex AI Docs])**

- **Source Information:** This technical documentation details the specific API request parameters for controlling generation in Vertex AI text models. It provides precise definitions for `topP` (selecting tokens until their cumulative probability sum reaches the threshold P) and `topK` (selecting from the K most probable tokens). It clarifies the typical interaction where Top-K/Top-P filters are applied before Temperature-based sampling.
- **Relevance to Article:** Provides the authoritative technical definitions [4] for the Top-K and Top-P sampling parameters explained conceptually in the article, including their interaction mechanism within the Vertex AI platform.

**5. Zero-Shot Learning Paper ([Ref 5: arXiv:2109.01652 - FLAN])**

- **Source Information:** This paper ("Finetuned Language Models Are Zero-Shot Learners") investigates _instruction tuning_ – fine-tuning LLMs on a diverse collection of NLP tasks presented via natural language instructions. The key finding is that this process significantly enhances the model's ability to perform _unseen_ tasks in a zero-shot setting (without task-specific examples in the prompt), often surpassing standard large models like GPT-3.
- **Relevance to Article:** While the article uses "zero-shot" in the common sense (no examples in the prompt), this paper [5] provides technical background on how the underlying zero-shot capabilities of models can be substantially improved through specific training methodologies like instruction tuning.

**6. Google Cloud Model Garden ([Ref 6: Vertex AI Docs])**

- **Source Information:** Model Garden is presented as a centralized library within Vertex AI for discovering, experimenting with, customizing, and deploying a wide range of foundation models (Google's and third-party/open-source), fine-tunable models, and task-specific solutions. It integrates with other Vertex AI tools like Studio and provides deployment infrastructure.
- **Relevance to Article:** This serves as the platform context [6] where users would likely access and manage the `gemini-pro` model used in the article's examples within the Google Cloud environment. _Note: The article incorrectly cites [6] for Step-Back prompting; [8] is the correct source._

**7. Few-Shot Learning Paper ([Ref 7: arXiv:2005.14165 - GPT-3])**

- **Source Information:** This seminal paper ("Language Models are Few-Shot Learners") demonstrated that scaling LLMs (like GPT-3 with 175B parameters) enables them to perform new tasks effectively when provided with only a few examples (shots) directly in the prompt, without requiring any model fine-tuning (gradient updates).
- **Relevance to Article:** This paper [7] established the concept and effectiveness of few-shot prompting, which the article highlights as a crucial best practice. It provides the theoretical and empirical foundation for the one-shot/few-shot techniques discussed.

**8. Step-Back Prompting Paper ([Ref 8: OpenReview])**

- **Source Information:** This paper ("Take a Step Back: Evoking Reasoning via Abstraction...") introduces the Step-Back Prompting technique. The methodology involves prompting the LLM to first abstract high-level concepts or principles from the specific details of a query, then using this abstracted understanding to guide the reasoning process for solving the original query. Experiments showed significant improvements on complex reasoning tasks.
- **Relevance to Article:** This is the originating research [8] for the Step-Back prompting technique explained and demonstrated with the FPS storyline example in the article.

**9. Chain of Thought (CoT) Paper ([Ref 9: arXiv:2201.11903])**

- **Source Information:** This paper ("Chain-of-Thought Prompting Elicits Reasoning...") introduced CoT prompting. The method involves providing few-shot examples that include intermediate reasoning steps leading to the final answer. It was shown to significantly improve LLM performance on arithmetic, commonsense, and symbolic reasoning tasks, particularly for larger models, by guiding them through a step-by-step process.
- **Relevance to Article:** This is the foundational paper [9] defining the Chain of Thought technique discussed extensively, including its methodology, benefits (reasoning improvement, interpretability), and the math problem example.

**10. GitHub CoT/ReAct Notebook 1 ([Ref 10: GitHub - GoogleCloudPlatform/generative-ai])**

- **Source Information:** Based on repository structure, this link likely points to a Jupyter/Colab notebook within the main Google Cloud generative AI examples repository. Such notebooks typically provide practical Python code demonstrating the implementation of techniques like Chain of Thought and ReAct, using Google Cloud services (Vertex AI) and potentially libraries like LangChain.
- **Relevance to Article:** Serves as a supplementary code resource [10] allowing practitioners to see concrete implementations of the CoT and ReAct concepts discussed theoretically.

**11. Self-Consistency Paper ([Ref 11: arXiv:2203.11171])**

- **Source Information:** This paper ("Self-Consistency Improves Chain of Thought Reasoning...") proposes Self-Consistency as an enhancement to CoT. The methodology involves sampling multiple diverse reasoning paths using non-greedy decoding (via temperature) for the same CoT prompt, and then selecting the most frequent final answer through marginalization/majority voting. It significantly improves results on reasoning benchmarks over standard greedy CoT.
- **Relevance to Article:** Provides the formal definition and methodology [11] for the Self-Consistency technique explained as a way to improve robustness by exploring multiple reasoning avenues.

**12. Tree of Thoughts (ToT) Paper ([Ref 12: arXiv:2305.10601])**

- **Source Information:** This paper ("Tree of Thoughts: Deliberate Problem Solving...") introduces the ToT framework. It generalizes CoT by allowing the LLM to explore multiple reasoning paths concurrently in a tree structure. It involves steps like thought generation, state evaluation (using value or voting heuristics), and search algorithms (BFS/DFS) to navigate potential solutions, enabling backtracking and deliberate exploration.
- **Relevance to Article:** This is the originating research [12] for the Tree of Thoughts concept, explaining its structure and advantage for complex, exploratory problem-solving.

**13. ReAct Paper ([Ref 13: arXiv:2210.03629])**

- **Source Information:** This paper ("ReAct: Synergizing Reasoning and Acting...") introduces the ReAct framework. The core idea is to interleave reasoning steps (`Thought`) with actions (`Act`) that interact with external tools/environments. The observations from actions inform subsequent reasoning. ReAct was shown to improve performance on knowledge-intensive QA and decision-making tasks by allowing models to retrieve external information or interact with environments.
- **Relevance to Article:** Defines the ReAct paradigm [13], explaining its thought-action-observation loop and its ability to combine internal reasoning with external tool use, as demonstrated in the Metallica example.

**14. GitHub CoT/ReAct Notebook 2 ([Ref 14: GitHub - applied-ai-engineering-samples])**

- **Source Information:** This link points to a specific Jupyter/Colab notebook within a repository focused on applied AI engineering samples on Google Cloud. This notebook explicitly demonstrates advanced prompting techniques, specifically Chain of Thought and ReAct, using Vertex AI (likely with Gemini models) and providing runnable Python code examples.
- **Relevance to Article:** Offers a targeted, practical code implementation resource [14] for users wanting to apply the CoT and ReAct techniques discussed, likely using the specific tools (Vertex AI, Gemini) featured in the article's examples.

**15. Automatic Prompt Engineering (APE) Paper ([Ref 15: arXiv:2211.01910])**

- **Source Information:** This paper ("Large Language Models are Human-Level Prompt Engineers") proposes the Automatic Prompt Engineer (APE) algorithm. The methodology involves using an LLM to generate instruction candidates for a task, evaluating these candidates (e.g., based on the target task performance achieved by another LLM using the candidate prompt), and selecting the highest-scoring instruction. It demonstrated that automatically discovered prompts could match or exceed human-engineered prompt performance.
- **Relevance to Article:** Provides the formal basis [15] for the Automatic Prompt Engineering technique, explaining its generate-evaluate-select methodology.

This detailed review incorporates the specific findings and methodologies from each referenced source, providing a technical foundation for the concepts and techniques presented in the original article.