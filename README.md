# 5-Day Gen AI Intensive Course

## Foundation - Transformer Architecture
Comes from a google project focused on language translation from 2017. 
For language translation they have a encoder taking in input, representing it. And the decoder uses the representation to make the translation. 

![image](https://github.com/user-attachments/assets/91a5ae53-6988-4503-8f62-0cee2b771892)

The input text needs to be prepped for the model. So, the text is turned into tokens based on a specific vocabulary the model uses. The token is then converted to a dense vector embedding that captures the meaning of the token. <br>
Transformers process all the tokens at the same time, so positional encoding is needed. The choice of positional encoding will affect how well the model understands longer sentences or longer sequences of characters. <br><br>

Self-attention - creates query key value vectors.
key is a label attached to each word it represents.
value is the actual information the word carries. 
Model calculates the scores and normalizes it to becomem attentions weights. It tells how each word should pay attention to the others. Uses those weights to create a weighted sum of all the characters. This representatiion takes into account the relationship of one word with every other word. All processed parallely. That is why the transformer architecture is so popular. 

Newer LLMs go with the decoder only architecture. Encoded generally not needed for conversations or text generations. Special type of self-attention - Masked self-attention. It can only see the text generated before, like real conversation. Simpler design.

Multi-Head attention 
Self attention several times at the same time, but with different sets of query, key and matrices. Each of these parallel self attention processes looks for different stuff - like grammar, meaning between words. Combines to get a deeper understanding of the words. 

Layer normalization - Helps to keep the activation levels at stable levels to make the training go faster and gives better results in the end. 
Residual connections - Act like shortcuts within the network. Let the original input to directly go to the output so that it can remember what it learned earlier even if it goes through many layers. Prevents vanishing gradient problem where the signal gets weaker and weaker as it goes deeper. 

Feed Forward layer
Feed Forward Network is applied to each token's representation seperately. Usually two linear transformation with a non linear transformation in between like relu, gelu. Gives the model more power to represent info and learn the complex info of the input. 


Making these models modre efficient - 
Experts are used for efficientcy depending on the task. 


After the first transformer paper. GPT1 from openai - 2018. Decoder only architecture, trained on a massive dataset of books called the books corpus. unsupervised pre-training. Patterns from raw-text and learning from that. Problems - Repetitive. Not for long conversations. 

Then came Google Bard. Understand conversations but couldnt hold conversations. 

GPT2 - Webtext datasets - 2019. Many more parameters in the model. Much better coherence, could handle longer dependencies between words, could learn new tasks without being trained on them - zero shot learning - learning from examples

GPT3 - 2020 - Billions of parameters. Few shot learning - Learning from a handful examples. Follow instructions written in natural language. 

GPT3.5 - Great at understanding and writing code.

GPT4 - Multimodal. Could handle text and images together. Context window size was huge. 

Google Lambda - 2021 - Designed for natural sounding converstations. 

DeepMind Gopher - 2021 - Using high quality data for training - Massive Text. Was good with knowledge tasks. But not with reasoning. Also found out that building a model with more parameters just wont help for every type of tasks. Some tasks need different approaches. 

Google Graham - Huge models ran much faster. Way less compute power but comparable performance to GPTS. Efficiency. 

Deepmind Chinchila 2022 - Given parameters need much larger datasets. 

Google Paulm(2022) and Paulm2(2023) - More about efficiency. 
Paulm2 foundation of the GenAi in google cloud

Google Gemini - Multimodal. Text Images Audio Video. Architectural improvements to scale the model really big. Optimized to run fast on TPUs (tensor processing units)
1.5 Pro - Fast. Huge context windows


OPEN SOURCE LLM 
Gemma Gemma2. 
Gemma2 2b parameter version that can run on a single gpu. 

Meta llama3

Mistral AI

OpenAI O1 models

Deepseek 

GrokAI from XAi

## LLM Training
1. Tons of data. Learning grammar vocab. Resource intensive - huge compute power
2. Fine tuning - Take the model and train it to a smaller specific dataset. Specific to the task we want it to do.
  SFT - Supervised Fine Tuning. Lots of examples of questions and the correct answers.
  RLHF - Reinforcement Learning from Human Feedback. Aligning output with what humans prefer. Reward model giving rewards for what the trainers like better.
  PEFT - Parameter Efficient Fine Tuning

## Prompt Engineering 
Designing the input in such a way that the output is as desired.
Zero Shot Training - Direct instruction or question with no example. 
Few Shot Prompting - With examples
Chain of thought Prompting - Show the model how to think through the problem step by step. 

## Sampling techniques
Can affect the quality, creativity, diversity of the output. Factual and Accurate or Creative and Imaginative. 
Types 
- Greedy - Always picks the next likely text. Fast but repetitive output
- Random sampling - Introduces more randomness, more creative output. Higher chance of getting nonsensical sense.
Temperature parameter to adjust randomness. Higher temp, more randomness.
- Top K sampling limits the model's choices to the top K most likely tokens which help control output.
- Top P sampling / nucleus sampling - Dynamic threshold based on the probabilities of the tokens.
- Best of N sampling generates multiple responses and picks the best ones based on some criteria.

## Evaluation of LLMs
Traditional metrics like accuracy or F1 score dont capture the whole picture for open-ended stuff like text generration
Evaluation needs to be multifaceted. Data specifically designed for the task being evaluated and must reflect what the model will see in the real world, including real user interactions.                        Consider the whole system and not just the model. 
Define what good means for the specific use case. May be accuracy, helpfulness, creativity, factual correctness, adherence to a certain style. 
Multifaceted
Human Evaluation required. 
LLM Powered auto raters - Generative models, reward models and discriminative models. Need to be calibrated. 

## Speeding up Inference
Optimizing for speed critical applications. 
Trade-offs. 
Quality of the output with speed/cost of generating. 
Latency of a single request and overall throughput of the system. 

1. Output approximating methods. Involves changing the output slightly to gain efficiency. 
Quantization - reducing the numerical precision of the weights and activations. Saves memory, makes calculations faster with a small reduction in accuracy. 
Quantization Aware Training QAT - Minimizes those accuracy reductions
Distillation - Training a smaller model to mimic a larger one. Student model is much faster and more efficient than the Teacher model while still achieving a good accuracy. Data distillation, knowledge distillation and on policy distillation
3. Output preserving methods. Keep output the same but optimize the computation.
Flash Attention is specifically designed to optimize the self attention calculations within the transformers. Minimizes the amount of data movement needed during the calculations, which can be a huge bottleneck. Output is exactly the same.
Prefix caching - For repeating parts of the input. Saves time. Caching the self-attention calculations for the initial parts of the inputs so it does not have to redo them for every turn.
Speculative decording - smaller, faster drafter model to poredict a bunch of future models. The main model checks and accepts the tokens if they are right and skips the calculations for them which speeds up the decoding process.
Batching - Multiple requests at the same time, more efficient than doing them one by one
Parallelizations - Splitting up the computation across multiple processors. 

# Prompt Engineering 
Designing the input in such a way that the output is as desired.
Output length - Token length directly impacts costs, processing time. Low token limit being set does not cut it. 

## Sampling controls
Temperature - Randomness. 
Low temperature - Most predictable output. Best guess. 
High temperaturem - More randomness, explore new ideas. Breakthroughs. 
- Top K sampling limits the model's choices to the top K most likely tokens which help control output.
- Top P sampling / nucleus sampling - Dynamic threshold based on the probabilities of the tokens.
- Best of N sampling generates multiple responses and picks the best ones based on some criteria.
When three of temp, Top K and Top P are on, Model puts together the words that meet the Top K and Top P criteria. Then the temperature setting comes in to pick from the set by probabilities.
Suggested Temp of 0.2, Top P of 0.95 and Top K of 30.
Creative Temp of 0.9, Top P of 0.99 and Top K of 40.
Factual Temp of 0.1, Top P of 0.9 and Top K of 20.
Single correct answer needed, best bet - Temperature of 0

Repition Loop Bug
Where the model gets stuck repeating the same words or phrases over and over. Happens both on low and high temps. Low temps it can be too predictable and get stuck in a loop revisiting the same words. High temps it can be so random that it revisits the words and gets stuck in a loop. 

Key is to fine tune the parameters to find the sweet spot where there is creative, interesting answers.

Prompt Engineering 
- Crafting clear prompts is of essence.
- Document the prompts and see what works best and why. 

General / Zero Shot prompting
One Shot Prompting
Few Shot Prompting

Quality and relevance of the chosen examples are crucial. Even small errors will lead to confusion, garbage output. 

For a high level guidance to the LLM
System Prompting - Setting overall context and purpose. Defining big picture. Output requirements... Using system prompts for output can help limit errors and ensure the data comes back in a usable order.
Contextual Prompting - Providing background information relevant to the task at hand. More context, more helpful and relevant the model's response is likely to be. 
Role Prompting - Giving LLM a persona/identity. Clear understanding of the perspective it needs to take. 
Stepback Prompting - Ask the LLM to a broader question before diving into the specific question. Insightful and creative responses. Helps mitigate biases in the LLMs responses. 

Chain of Thought Prompting - Boosting the model's reasoning capabilities. Prompting it to generate intermediate reasoning steps before giving the final answer or code suggestion. More tokens, higher cost and processing time. 

Self Consistency - Get the model to generate multiple reasoning paths for the same prompts and then we choose the most consistent answer. 

Tree of Thoughts - Multiple reasoning paths simultaneously. Exploring and backtracking. For very challenging problems.

React - Reason and Act 
Using the ability of the LLM to reason and use tools such as search engine, code interpreters, apis. 
