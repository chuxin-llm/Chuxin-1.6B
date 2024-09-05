# Chuxin-1.6B-Base
<br>

## 介绍 (Introduction)

[**Chuxin-1.6B-Base**](https://huggingface.co/chuxin-llm/Chuxin-1.6B-Base)是16亿参数规模的模型。Chuxin-1.6B完全基于开源数据构建，在经过超大规模数据训练后，Chuxin-1.6B在各类下游任务上具有非常强的竞争力。

[**Chuxin-1.6B-1M**](https://huggingface.co/chuxin-llm/Chuxin-1.6B-1M)是基于Chuxin-1.6B-base模型在1M窗口下训练后的结果，大海捞针实验显示其具有非常强的上下文检索能力。


如果您想了解更多关于Chuxin-1.6B开源模型的细节，我们建议您参阅我们的[技术报告](https://arxiv.org/pdf/2405.04828)

[**Chuxin-1.6B-Base**](https://huggingface.co/chuxin-llm/Chuxin-1.6B-Base) is a model with 1.6 billion parameters. Chuxin-1.6B is built entirely on open-source data. After being trained with large-scale data, Chuxin has very competitive capabilities in various downstream tasks.

[**Chuxin-1.6B-1M**](https://huggingface.co/chuxin-llm/Chuxin-1.6B-1M) is the result of training the Chuxin-1.6B-base model with a 1M windows. Experiments such as searching for a needle in a haystack demonstrate its strong contextual retrieval abilities.

If you would like to learn more about the Chuxin-1.6B open-source model, we suggest you refer to our [technical report](https://arxiv.org/pdf/2405.04828).
<br>

## 快速使用（Quickstart）

您可以通过以下代码轻松调用：

You can easily call the model with the following code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("chuxin-llm/Chuxin-1.6B-Base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("chuxin-llm/Chuxin-1.6B-Base", device_map="auto", trust_remote_code=True, bf16=True).eval()
inputs = tokenizer('蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs, max_new_tokens=15, do_sample=False)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# 蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是亚的斯亚贝巴（Addis Ababa）...
```

## 评测效果（Evaluation）

### (常识推理和阅读理解)  Common Sense Reasoning and Reading Comprehension tasks

| Model         | size | ARC-c |ARC-e |Boolq |Copa |Hellaswag |OpenbookQA |Piqa |Sciq |Winogrande |Avg|
|:--------------|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| Gemma     |    2B   |     48.98    |  78.45    |  69.51   |  84    |  71.73     |  39.8    |  78.02   |  94.3    |  65.51    |  70.03      |
| H2O-Danube†     |    1.8B  |     35.84   |  62.29   |  65.81   |  -   |  68.20     |  37.6    |  76.93   |  -    |  61.96   |  -     |
| Qwen1.5    |    1.8B   |      37.03     |  67.51    |  66.64    |  78    |  61.60     |  34.40    |  73.99     |  93     |  61.56     |  63.74    | 
| StableLM 2     |    1.6B    |      43.52   |69.44     | 75.5    | 84     | 70.3      | 39.6     | 76.82    | 96.1      | 64.17     | 68.82     |
| OpenLlama†   |    3B    |      34   |69| 68| -| 49| 40| 75| -| 62 |-|
| CT-LLM |  2B |  34.81   |  65.49   | 62.45    | 74    | 54.77      | 33.4     | 71.38   | 90.6     | 57.85     | 60.63     | 
| TinyLLama |  1.1B  |  34.81   | 67.47     | 63.15      | 74     | 60    | 34.6      | 73.12     | 88.8     | 58.88     | 61.64    |
| OLMo |  1B |  34.22   | 67.55     | 61.4     | 82     | 63.96      | 36.4     | 75.1    | 86.7     | 60.3      | 63.07     |
| Chuxin-1.6B-Base |  1.6B |  39.68  | 71.38     | 71.25      | 83    | 66.09     | 35.00      | 77.09     | 95     | 63.54      | 66.89     |

带有†的模型表示我们直接报告了相应论文中的分数，其他的则来自于我们重新测试的结果。

Models with † denote that we directly report the scores from the corresponding paper, and others are from our implementation.

### Open LLM LeaderBoard

| Model         | size | ARC-c  |HellaSwag|MMLU |TruthfulQA |Winogrande |GSM-8k |Avg |Avg wo GSM|
|:--------------|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| Gemma     |    2B   |     48.98   |  71.73     |  42.47    |  33   |  65.51    |10.08|  45.3    |  52.34      |
| H2O-Danube    |    1.8B    |      39. 68    | 69.75    | 25.97    | 33.63   | 64.17| 2.05    | 39.21     |46.64|
| Qwen1.5†     |    1.8B   |      37.88     |   61.42    |  46.71   |  39.43    |  60.3     |  33.59     |  46.55   | 49.15|
| StableLM 2     |    1.6B    |      43.52 |70.3  | 39.8     | 36.61     | 64.17   | 17.29      | 45.28     | 50.88    |
| OpenLlama†      |     3B    |    39.9  | 71.6    | 27.1    | 34.8     | 67   | 0.9      |40.3|48.08|
| CT-LLM |  2B |  34.81   | 54.77      | 37.81     | 39.81   | 57.85     | 7.35     | 38.73     | 45.01|
| TinyLLama |  1.1B  |  33.87  | 60.31   | 26.04     | 37.32   | 59.51    | 1.44     | 36.42    |43.41|
| OLMo |  1B |  34.22   | 63.96      | 35.44     | 35.53    | 62.67     | 9.86     | 41.81    |48.2|
| Chuxin-1.6B-Base |  1.6B |  39.68  | 66.09     | 41.07      | 37.65    | 63.54    | 12.66     | 43.45    |49.61|

带有†的模型表示我们直接报告 Open LLM排行榜的分数，其他的则来自于我们重新测试的结果。

Models with † denote that we directly report the scores from the Open LLM Leaderboard, and others are from our implementation.

### CMMLU, C-Eval and HumanEval

| Model         | size | C-Eval  |CMMLU|HUMANEVAL |
|:--------------|:----------:|:-----------:|:-----------:|:-----------:|
| Gemma     |    2B   |     31   |  31.06    |  9.51| 
| Qwen1.5    |    1.8B   |      59.38     |   57.08  |  23.17   | 
| StableLM 2     |    1.6B    |      29.27 |30.1 | 7.32     | 
| CT-LLM |  2B |  36.78  | 36.4      | 9.15     | 
| Chuxin-1.6B-Base |  1.6B |  39.31  | 37.11     | 9.76 |


## 评测效果-长文（Evaluation-Looooong）

### 常识推理和阅读理解 (Common Sense Reasoning and Reading Comprehension tasks)

| Model         | size | ARC-c |ARC-e |Boolq |Copa |Hellaswag |OpenbookQA |Piqa |Sciq |Winogrande |Avg|
|:--------------|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| chuxin-1.6B-base |  1.6B |  39.68  | 71.38     | 71.25      | 83    | 66.09     | 35.00      | 77.09     | 95     | 63.54      | 66.89     |
| chuxin-1.6B-32k |  1.6B |  39.16  | 70.66     | 67.71     | 81   | 65.69     | 35.8      | 76.88    | 94.2    | 62.51     | 65.96     |
| chuxin-1.6B-64k |  1.6B |  38.48  | 70.24     | 67.52     | 82    | 65.6     | 35.2      | 76.61     | 94.3    | 63.3      | 65.92     |
| chuxin-1.6B-128k |  1.6B |  39.08  | 69.4     | 67.71      | 80    | 65.74    | 35.4      | 76.39    | 94.1    | 63.3      | 65.68     |
| chuxin-1.6B-256k |  1.6B |  40.19  | 70.75     | 69.3      | 78    | 65.85    | 35.8     | 76.88    | 93.5     | 63.85     | 66.01    |
| chuxin-1.6B-512k |  1.6B | 40.61 |71.21| 67.77 |78| 64.82| 34.8| 76.88| 93.6| 61.88| 65.51|
| chuxin-1.6B-1M |  1.6B | 41.13| 72.26| 62.08| 75| 64.59 |34.8| 76.71| 93.33| 62.43| 64.7|

### Open LLM LeaderBoard

| Model         | size | ARC-c  |HellaSwag|MMLU |TruthfulQA |Winogrande |GSM-8k |Avg |Avg wo GSM|
|:--------------|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| chuxin-1.6B-base |  1.6B |  39.68  | 66.09     | 41.07      | 37.65    | 63.54    | 12.66     | 43.45    |49.61|
| chuxin-1.6B-32k |  1.6B |  39.16 | 65.69     | 38.63      | 35.66    | 62.51     | 11.6     | 42.21    | 48.33|
| chuxin-1.6B-64k |  1.6B |  38.48  | 65.6    | 38.43     | 35.07    | 63.3     | 11.9      | 42.13|48.18|
| chuxin-1.6B-128k |  1.6B |  39.08  | 65.74    | 37.65     | 34.89    | 63.3    | 11.07     | 41.96|48.13|
| chuxin-1.6B-256k |  1.6B |  40.19  | 65.85     | 37.16      | 35.2    | 63.85     | 10.16      | 42.07    |48.45|
| chuxin-1.6B-512k |  1.6B |  40.61| 64.82| 36.66| 33.66| 61.88| 8.11| 40.96| 47.53|
| Chuxin-1.6B-1M |  1.6B |  41.13 |64.59| 35.76| 34.67| 62.43| 6.82| 40.9| 47.72|


## 引用 (Citation)

如果你觉得我们的工作对你有帮助，欢迎引用！

If you find our work helpful, feel free to give us a cite.

```
@article{chuxin,
  title={CHUXIN: 1.6B TECHNICAL REPORT},
  author={Xiaomin Zhuang, Yufan Jiang, Qiaozhi He, Zhihua Wu},
  journal={arXiv preprint arXiv:2405.04828},
  year={2024}
}
```
<br>
