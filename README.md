# Infusing Wikipedia Knowledge to Enhance Stance Detection

his repo is the implemention of our [paper](https://arxiv.org/abs/2204.03839) "Infusing Wikipedia Knowledge to Enhance Stance Detection", where we propose to utilize the background knowledge from Wikipedia about the target to improve stance detection.


## Dataset Preparation
In this paper, we experiment on three datasets: [PStance](https://aclanthology.org/2021.findings-acl.208/), [COVID19-Stance](https://aclanthology.org/2021.acl-long.127/), and [VAST](https://aclanthology.org/2020.emnlp-main.717.pdf). 

1. <em>VAST</em> is publicly available at [here](https://github.com/emilyallaway/zero-shot-stance/tree/master/data/VAST) and thus the data is also included in this repo.
2. The authors of <em>PStance</em> did not make the dataset readily accessible on the Internet. To gain access to it, please contact the first author of the paper. After you have the data files (<em>raw_{phase}_{target}.csv</em>, <em>phase</em> $\in$ {<em>train</em>, <em>val</em>, <em>test</em>}, <em>target</em> $\in$ {<em>bernie</em>, <em>trump</em>, <em>biden</em>}), put them under <em>data/pstance</em> and run the jupyter notebook to pre-process the data. 
3. For <em>COVID19-Stance</em>, the author just made the tweet ids publicly available at [here](https://github.com/kglandt/stance-detection-in-covid-19-tweets/tree/main/dataset). To gain the tweet contents, you can either use Twitter API or contact the first author. After you have the data files, put them under <em>data/covid19-stance</em>


## Installation
Install [Pytorch](https://pytorch.org/get-started/locally/) and [Huggingface Transformers](https://huggingface.co/docs/transformers/installation).

## Run
PStance, target-specific stance detection, Biden
```angular2html
python run_pstance_biden.py
```

COVID19-Stance, target-specific stance detection, face mask
```angular2html
python run_covid_fauci.py
```

PStance, cross-target stance detection, Biden $\rightarrow$ Sanders
```angular2html
python run_pstance_biden2sanders.py
```


VAST, zero/few-shot stance detection
```angular2html
python run_vast.py
```


## Citation
```angular2html
@inproceedings{he2022infusing,
  title={Infusing Knowledge from Wikipedia to Enhance Stance Detection},
  author={He, Zihao and Mokhberian, Negar and Lerman, Kristina},
  booktitle={Proceedings of the 12th Workshop on Computational Approaches to Subjectivity, Sentiment \& Social Media Analysis},
  pages={71--77},
  year={2022}
}
```


