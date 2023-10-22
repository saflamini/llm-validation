## A Collection of Resources on Reducing Hallucinations

### UPDATE Oct 21, 2023
- This is a work in progress. Below you can find field notes on reducing hallucinations & other helpful resources.

#### What is a hallucination

I like this definition from [A Survey of Hallucination in Large Foundation Models]('https://arxiv.org/abs/2309.05922'):



## Tactics to Improve LLM Reasoning 

#### Chain of Thought Prompting
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf)
TL;DR - generating a 'chain of thought' or a 'series of intermediate reasoning steps' improves LLM performance.
The paper specifies a tactic for providing reasoning 'examples' as a way to improve overall performance.

The intuition here is that, when a human performs a complex task, they often find it helpful to 'decompose' the task down into smaller steps, and reason through them one at a time. LLMs are next token predictors, and they'll use their *own* generated tokens as inputs for generating the next tokens. If the LLM is asked to generate high quality, step by step reasoning before providing you with an answer, it's more likely that the final answer will be of high quality because it will use its own high level reasoning as input for that final answer.

Asking the model to 'think step by step' often is considered a form of chain of thought prompting, but the original paper authors seem to believe that it's ideal to provide few shot reasoning examples. For arithmetic reasoning examples, the authors included a prompt which showcased 12 Q&A samples that demonstrated COT reasoning before injecting a user query into the end of the prompt.
