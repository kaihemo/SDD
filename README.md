<div align="center">
 👋 Hi, everyone! 
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/93481cda-a7f3-47f3-b333-fe6b3da86b78">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

<!-- 注释：以上为Seed官方信息，可直接复制使用，请注意导入“Seed WeChat”（第12行）、“Seed logo”(第20行)图片替换 -->


# Scale-Distribution Decoupling (SDD)
<p align="center">
  <a href="https://github.com/kaihemo/SDD/tree/sdd">
    <img src="https://img.shields.io/badge/SDD-Project Page-yellow"></a>
  <a href="https://arxiv.org/pdf/2502.15499">
    <img src="https://img.shields.io/badge/SDD-Tech Report-red"></a>
  <!-- <a href="XXXX">
    <img src="https://img.shields.io/badge/SDD-Hugging Face-orange"></a> -->
  <a href="[XXX](https://opensource.org/license/MIT)">
    <img src="https://img.shields.io/badge/License-MIT-blue"></a>
</p>

This repository contains the official implementation of **Scale-Distribution Decoupling (SDD)**, a novel method developed to stabilize the training of large language models (LLMs) by effectively preventing **gradient explosion and dissipation**. By decoupling the scale and distribution of the fully connected layer weights, SDD not only enhances training robustness but also accelerates convergence and improves optimization efficiency. These benefits are particularly evident in deep Transformer-based models, especially in postnorm-based configurations.
<div align="center">
  <img src="./figures/SDD.jpg" width="840px" />
  <p></p>
</div>

## News
- [02/18/25] Code release for Scale-Distribution Decoupling (SDD) applied to LLMs.
- [02/18/25] Paper temporarily released on arXiv.

## Table of Contents
- [Scale-Distribution Decoupling (SDD)](#scale-distribution-decoupling-sdd)
  - [News](#news)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
    - [Installation](#installation)
    - [Datasets](#datasets)
    - [Training](#training)
    - [Usage in Other Frameworks](#usage-in-other-frameworks)
  - [License](#license)
  - [Citing this work](#citing-this-work)
  - [Acknowledgement](#acknowledgement)
  - [About ByteDance Seed Team](#about-bytedance-seed-team)

## Usage
This repository contains the implementation of Scale-Distribution Decoupling (SDD) for stabilizing and improving the training of large language models. The main components of the repository are organized into directories as follows:
- [dense/](/dense/): Contains implementations of dense models used in experiments.
- [moe/](moe/): Implements the Mixture of Experts (MoE) models used in experiments.


### Installation
To install the necessary dependencies, run the following command:
```bash
python3 -m pip install -r requirements.txt
```

### Datasets

We use the **OLMoE-mix-0924** dataset, which can be downloaded from [here](https://huggingface.co/datasets/allenai/OLMoE-mix-0924). After downloading, use the following command to convert the dataset into the format required by our model:
```python
dolma tokens \
--documents ${PATH_TO_DOWNLOADED_DATA} \
--destination ${PATH_WHERE_TO_SAVE_TOKENIZED_DATA} \
--tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
--max_size '2_147_483_648' \
--seed 0 \
--tokenizer.eos_token_id 50279 \
--tokenizer.pad_token_id 1 \
--processes ${NUMBER_OF_CPU_CORES_TO_USE}
```

### Training
To train a large language model using Scale-Distribution Decoupling, execute the following command:
```bash
cd dense    # for dense experiments
# cd moe    # for MoE experiments
bash run.sh
```
Training metrics, including loss, accuracy, and other relevant indicators, will be displayed and tracked via wandb-service.

For ease of understanding and comparison, the SDD model and the baseline model are encapsulated in two separate branches. You can use git diff to inspect the differences between the two models:
- To switch to the baseline model branch, run:
    ```bash
    git checkout baseline
    ```
- To switch to the SDD model branch, run:
    ```bash
    git checkout sdd
    ```
- To directly compare the implementations of SDD and the baseline:
    ```bash
    git diff sdd baseline
    ```

### Usage in Other Frameworks
To replace ***all*** Linear operations in your models, use the following **SDDLinear** class. This class combines the linear transformation with RMSNorm to stabilize training.
```python
class SDDLinear(torch.nn.Module):
    def __init__(self, input_hidden_size, output_hidden_size, bias=False, eps=1e-6):
        super(SDDLinear, self).__init__()
        self.linear = torch.nn.Linear(
            in_features=input_hidden_size,
            out_features=output_hidden_size,
            bias=bias)
        
        self.norm = torch.nn.RMSNorm(
            normalized_shape=output_hidden_size,
            eps=eps,
            elementwise_affine=True)

    def forward(self, x):
        return self.norm(self.linear(x))
```

## License
This project is licensed under MIT. See the [MIT License](https://opensource.org/license/MIT) flie for details.

## Citing this work
If you find this work helpful or use it in your research, please consider citing our paper:
```bibtex
@article{wang2025scale,
  title={Scale-Distribution Decoupling: Enabling Stable and Effective Training of Large Language Models},
  author={Wang, Ya and Zhuo, Zhijian and Zeng, Yutao and Zhou, Xun and Yang, Jian and Li, Xiaoqing},
  journal={arXiv preprint arXiv:2502.15499},
  year={2025}
}
```

## Acknowledgement
Our work is based on the following repositories:
- [OLMo](https://github.com/allenai/OLMo)
- [OLMoE](https://github.com/allenai/OLMoE)


## About [ByteDance Seed Team](https://team.doubao.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.
