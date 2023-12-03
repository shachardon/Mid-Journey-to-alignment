# Human Learning by Model Feedback: The Dynamics of Iterative Prompting with Midjourney

<p align="center">
  <img src="assets/fig_1.jpeg" width=512px>
</p>

The language people use when they interact with each other changes over the course of the conversation, as people dynamically adapt to each other. 

Will we see a systematic language change along the interaction of human users with a text-to-image model too? 

Generating images with a Text-to-Image model often requires multiple trials, where human users iteratively update their prompt based on feedback, namely the output image. Taking inspiration from cognitive work on reference games and dialogue alignment, we analyze the dynamics of the user prompts along such iterations. We compile a dataset of iterative interactions of human users with Midjourney. 

Paper link: http://arxiv.org/abs/2311.12131

---

Data
---
The dataset that was collected and used in this paper is available in the `data` folder.

The data is in a csv format, divided into 9 files (threads_i.csv for i in range(0, 9, 20000)).
It is also available as a Huggingface dataset [here][hf_data]

[hf_data]: https://huggingface.co/datasets/shachardon/midjourney-threads "markdown huggingface_dataset"

---

Code
---
The code for preparing the data is in the `prepare` folder.

---

Citiation
---
If you find this work useful, please cite our paper:

```
@misc{donyehiya2023human,
      title={Human Learning by Model Feedback: The Dynamics of Iterative Prompting with Midjourney}, 
      author={Shachar Don-Yehiya and Leshem Choshen and Omri Abend},
      year={2023},
      eprint={2311.12131},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
