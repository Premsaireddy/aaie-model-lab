**Phi2 Model Research**  

**Overview:** 

The Phi family of models released by Microsoft comprises small language models trained  on  high-quality  synthetic  and  web  data.  **Phi-2**,  a  2.7-billion-parameter transformer, is particularly notable because it matches or outperforms models several times  its  size  on  reasoning,  language  understanding,  math  and  coding  tasks microsoft.com. This report summarises available performance data for Phi-2 and outlines how it can be evaluated for the AAIE project. 

**Key facts about Phi-2:** 

- **Model size and training:** Phi-2 contains 2.7B parameters and is trained as a base (causal) language model with a next-word prediction objective. It is trained on  1.4 trillion  tokens  of  synthetic  and  web  data  curated  to  provide “textbook-quality” knowledge. The model was trained for 14 days on 96 A100 GPUs microsoft.com. 
- **Performance:** On benchmarks such as BigBench Hard (BBH), common-sense reasoning (PIQA, WinoGrande, ARC), language understanding (HellaSwag, OpenBookQA, MMLU), math (GSM8k) and coding (HumanEval, MBPP), Phi-2 matches  or  surpasses  larger  open-source  models  (e.g.,  Mistral-7B, Llama-2-13B)  microsoft.com. The  model’s  small  size  makes  it  suitable  for experimentation and fine-tuning. 
- **Safety:** Although Phi-2 has not undergone reinforcement learning from human feedback (RLHF), Microsoft’s researchers note that its safety profile (measured via scaled perplexity on ToxiGen) surpasses many aligned models because the training data emphasises high-quality content microsoft.com. 

**Advantages of Phi-2:** 

- **Small but capable:** 2–3B params with surprisingly strong reasoning vs. size— great quality/compute trade-off.** 
- **Runs on modest hardware:** Works well with 4-/8-bit quantization and feasible on a single consumer GPU.** 
- **Fast  iteration:**  Quick  to  load/infer  for  prototyping  prompts,  metrics,  and pipelines. 
- **Base Model behaviour:** Good for research tasks like perplexity based AI detection.** 

**Limitations of Phi2:** 

- Needs careful prompting or fine-tuning for polished, teacher-ready feedback. 
- For complex writing/long reasoning, 7B–13B+ instruct models often outperform it. 
- Less  room  for  very  long  essays/rubrics  in  a  single  pass  (workarounds: chunking/Sliding PPL). 
- Safety, style, and bias controls must be added in prompts/post-processing. 
- As a strong Language Model, it can assign low perplexity to both AI and well- written human text requires threshold calibration and auxiliary signals. 

**Takeaway for AAIE Project:** 

- Use Phi-2 only for lightweight AI-detection baselines (perplexity + calibrated thresholds), prompt experiments. 
