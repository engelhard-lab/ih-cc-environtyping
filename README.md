# Infinite-hierarchical-contrastive-clustering-for-personal-digital-envirotyping

![image](https://github.com/user-attachments/assets/3c490219-587c-40cc-be4b-8d638eba9751)
This repository contains the implementation of [Infinite Hierarchical Contrastive Clustering for Personal Digital
Envirotyping](https://arxiv.org/pdf/2505.15022), which builds on the contrastive clustering framework originally introduced in [Contrastive Clustering by Yunfan Li](https://github.com/Yunfan-Li/Contrastive-Clustering).

# Introduction
The influence of daily environments on human health and behavior has long been recognized. Characterizing  individual daily environments,  which we call **envirotyping**, and understanding the relationships between the envirotypes and health is essential to develop environment-aware intervention strategies.   
Current technologies allow for the easy collection of daily environment images, but a scalable and effective approach to identify personal, distinct environments remains lacking. While pre-trained image classification models excel at  identifying common envirotypes, they struggle with personal, distinct envirotypes.  
Motivated by the observation that the data augmentation process in contrastive learning mirrors the variability within repeatedly sampling of environmental images, we proposed a contrastive-learning-based clustering method to:
* Grouping images into distinct environment, and
* Grouping multiple distinct environment into clusters with similar environmental  features. 



# Usage

### Configuration
The configuration file, `config/config.yaml`, allows users to customize training.
### Train
Once setting up the training configuration, simply run `python train.py` to start model training process. 
### Cluster
After model training, using the same `config.yaml` used for the training for cluster inferencing. 

# Dataset
The data source includes images and associated health outcomes collected from a larger group of participants through a photo EMA process. This data was used to explore the relationship between daily environments and health outcomes. Due to the lack of written consent from participants for public data sharing, these data cannot be made available for public access.

