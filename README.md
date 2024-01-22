# FATE-ViT: Federated Vision Transformers
FATE-ViT is built on top of FATE-LLM with a focus on vision instead of NLP.

### Abstract
Vision Transformers (ViTs) have demonstrated remarkable performance improvements over conventional computer vision methods like Faster R-CNN and Fully Convolutional Networks (FCNs) in tasks such as Object Detection and Recognition. However, these large vision models face significant challenges when federated, primarily due to their substantial size. Training high-quality large vision models often necessitates vast amounts of data and is computationally intensive. These challenges have hindered their adoption, particularly by small and medium-sized enterprises with limited computational resources. In this repository, we present FATE-ViT, an extension of the FATE-LLM framework tailored for computer vision. FATE-ViT allows for efficient training of vision transformers by using a Low Rank Adaptation (LoRA) similar to FATE-LLM. FATE-ViT addresses the issues of model size and data requirements for image transformers, making it more accessible for a wider range of applications in the field of computer vision. FATE-ViT offers a comprehensive solution for federated learning of large vision models, enabling efficient training and preserving data privacy while reducing computational demands.

### Standalone deployment
Please refer to [FATE-Standalone deployment](https://github.com/FederatedAI/FATE#standalone-deployment).  
FATE-ViT is tested on FATE 1.11.2. Copy directory `python/fate_llm` to `{fate_install}/fate/python/fate_llm`

### Cluster deployment
Use [FATE-LLM deployment packages](https://github.com/FederatedAI/FATE/wiki/Download#llm%E9%83%A8%E7%BD%B2%E5%8C%85) to deploy,  refer to [FATE-Cluster deployment](https://github.com/FederatedAI/FATE#cluster-deployment) for more deployment details.

## Quick Start
- [Federated ViT CIFAR-10 tutorial](./doc/tutorial/parameter_efficient_llm/ViT-example.ipynb)
- [Federated DETR COCO tutorial](./doc/tutorial/parameter_efficient_llm/DETR.ipynb)
