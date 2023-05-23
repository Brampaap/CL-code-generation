# CL-code-generation
This repository contains experiments on training text2sparql(sql) models using curriculum learning methods

# Core idea
The data must be sorted by difficulty based on certain heuristics, the presented solution implements the root, baby step and linear schedulers.
Then the batch for each new iteration is saturated with more complex samples, thereby imitating the process of human learning - from simple to complex.

# Before run
Please make sure, you've setted up environmental variable.

```export PROJECT_PATH=<PATH_TO_PROJECT>```

# Metric
|                  | Exact match |
|------------------|-------------|
| No CL            | 0.5906      |
| Linear Scheduler | 0.6399      |
| Root Scheduler   | 0.6350      |

# References
1. [A Survey on Curriculum Learning](https://arxiv.org/pdf/2010.13166.pdf)
2. [PyTorch Lightning](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjFhriAoIv_AhVxkosKHWNAC5QQFnoECAoQAQ&url=https%3A%2F%2Fwww.pytorchlightning.ai%2Findex.html&usg=AOvVaw0dUoDetysXV3vO9AWfPtyy)
