# MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.7%20|%203.8%20|%203.9-blue.svg?style=&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mevis-a-large-scale-benchmark-for-video/referring-video-object-segmentation-on-mevis)](https://paperswithcode.com/sota/referring-video-object-segmentation-on-mevis?p=mevis-a-large-scale-benchmark-for-video)

**[🏠[Project page]](https://henghuiding.github.io/MeViS/)** &emsp; **[📄[arXiv]](https://arxiv.org/abs/2308.08544)**  &emsp; **[📄[PDF]](https://drive.google.com/file/d/1WRanGRaYPpaNfrwq4xRq0sfmiJLSr9-b/view?usp=sharing)** &emsp; **[🔥[Dataset Download]](https://codalab.lisn.upsaclay.fr/competitions/15094)** &emsp; **[🔥[Evaluation Server]](https://codalab.lisn.upsaclay.fr/competitions/15094)**

This repository contains code for **ICCV2023** paper:
> [MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions](https://arxiv.org/abs/2308.08544)  
> Henghui Ding,  Chang Liu,  Shuting He,  Xudong Jiang,  Chen Change Loy  
> ICCV 2023

<table border=1 frame=void>
  <tr>
    <td><img src="https://github.com/henghuiding/MeViS/blob/page/GIF/bird.gif" width="245"></td>
    <td><img src="https://github.com/henghuiding/MeViS/blob/page/GIF/Cat.gif" width="245"></td>
    <td><img src="https://github.com/henghuiding/MeViS/blob/page/GIF/coin.gif" width="245"></td>
  </tr>
</table>

### Abstract

This work strives for motion expressions guided video segmentation, which focuses on segmenting objects in video content based on a sentence describing the motion of the objects. Existing referring video object segmentation datasets downplay the importance of motion in video content for language-guided video object segmentation. To investigate the feasibility of using motion expressions to ground and segment objects in videos, we propose a large-scale dataset called MeViS, which contains numerous motion expressions to indicate target objects in complex environments. The goal of MeViS benchmark is to provide a platform that enables the development of effective language-guided video segmentation algorithms that leverage motion expressions as a primary cue for object segmentation in complex video scenes.

<div align="center">
  <img src="https://github.com/henghuiding/MeViS/blob/page/static/DemoImages/teaser.png?raw=true" width="100%" height="100%"/>
</div>
<p style="text-align:justify; text-justify:inter-ideograph;width:100%">Figure 1. Examples of video clips from <b>M</b>otion <b>e</b>xpressions <b>Vi</b>deo <b>S</b>egmentation (<b>MeViS</b>) are provided to illustrate the dataset's nature and complexity. <font color="#FF6403">The expressions in MeViS primarily focus on motion attributes and the referred target objects that cannot be identified by examining a single frame solely</font>. For instance, the first example features three parrots with similar appearances, and the target object is identified as <i>"The bird flying away"</i>. This object can only be recognized by capturing its motion throughout the video.</p>




<table border="0.6">
<div align="center">
<caption><b>TABLE 1. Scale comparison between MeViS and existing language-guided video segmentation datasets.
</div>
<tbody>
    <tr>
        <th align="right" bgcolor="BBBBBB">Dataset</th>
        <th align="center" bgcolor="BBBBBB">Pub.&Year</th>
        <th align="center" bgcolor="BBBBBB">Videos</th>
        <th align="center" bgcolor="BBBBBB">Object</th>
        <th align="center" bgcolor="BBBBBB">Expression</th>
        <th align="center" bgcolor="BBBBBB">Mask</th>
        <th align="center" bgcolor="BBBBBB">Obj/Video</th>
        <th align="center" bgcolor="BBBBBB">Obj/Expn</th>
        <th align="center" bgcolor="BBBBBB">Target</th>
    </tr>
    <tr>
      <td align="right"><a href="https://kgavrilyuk.github.io/publication/actor_action/" target="_blank">A2D&nbsp;Sentence</a></td>
      <td align="center">CVPR&nbsp;2018</td>
      <td align="center">3,782</td>
      <td align="center">4,825</td>
      <td align="center">6,656</td>
      <td align="center">58k</td>
      <td align="center">1.28</td>
      <td align="center">1</td>
      <td align="center">Actor</td>
    </tr>
    <tr>
      <td align="right" bgcolor="ECECEC"><a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/video-segmentation/video-object-segmentation-with-language-referring-expressions" target="_blank">DAVIS17-RVOS</a></td>
      <td align="center" bgcolor="ECECEC">ACCV&nbsp;2018</td>
      <td align="center" bgcolor="ECECEC">90</td>
      <td align="center" bgcolor="ECECEC">205</td>
      <td align="center" bgcolor="ECECEC">205</td>
      <td align="center" bgcolor="ECECEC">13.5k</td>
      <td align="center" bgcolor="ECECEC">2.27</td>
      <td align="center" bgcolor="ECECEC">1</td>
      <td align="center" bgcolor="ECECEC">Object</td>
    </tr>
    <tr>
      <td align="right"><a href="https://youtube-vos.org/dataset/rvos/" target="_blank">ReferYoutubeVOS</a></td>
      <td align="center">ECCV&nbsp;2020</td>
      <td align="center">3,978</td>
      <td align="center">7,451</td>
      <td align="center">15,009</td>
      <td align="center">131k</td>
      <td align="center">1.86</td>
      <td align="center">1</td>
      <td align="center">Object</td>
    </tr>
    <tr>
      <td align="right" bgcolor="E5E5E5"><b>MeViS (ours)</b></td>
      <td align="center" bgcolor="E5E5E5"><b>ICCV&nbsp;2023</b></td>
      <td align="center" bgcolor="E5E5E5"><b>2,006</b></td>
      <td align="center" bgcolor="E5E5E5"><b>8,171</b></td>
      <td align="center" bgcolor="E5E5E5"><b>28,570</b></td>
      <td align="center" bgcolor="E5E5E5"><b>443k</b></td>
      <td align="center" bgcolor="E5E5E5"><b>4.28</b></td>
      <td align="center" bgcolor="E5E5E5"><b>1.59</b></td>
      <td align="center" bgcolor="E5E5E5"><b>Object(s)</b></td>
    </tr>
  </tbody>
  <colgroup>
    <col>
    <col>
    <col>
    <col>
    <col>
    <col>
    <col>
    <col>
    <col>
  </colgroup>
</table>

## MeViS Dataset Download

⬇️ [Download the dataset from ️here☁️](https://codalab.lisn.upsaclay.fr/competitions/15094). 


**Dataset Split**
* 2,006 videos & 28,570 sentences in total;
* **Train set:** 1662 videos & 23,051 sentences, used for training;
* **Val<sup>u</sup> set:** 50 videos & 793 sentences, used for offline evaluation (e.g., ablation study) by users during training;
* **Val set:** 140 videos & 2,236 sentences, used for [**CodaLab online evaluation**](https://codalab.lisn.upsaclay.fr/competitions/15094);
* **Test set:** 154 videos & 2,490 sentences (not released yet), used for evaluation during the competition periods;
It is suggested to report the results on **Val<sup>u</sup> set** and **Val set**.



## Online Evaluation

Please submit your results of **Val set** on 
 - 💯 [**CodaLab**](https://codalab.lisn.upsaclay.fr/competitions/15094).

It is strongly suggested to first evaluate your model locally using the **Val<sup>u</sup>** set before submitting your results of the **Val** to the online evaluation system.

## File Structure

The dataset follows a similar structure as [Refer-YouTube-VOS](https://youtube-vos.org/dataset/rvos/). Each split of the dataset consists of three parts: `JPEGImages`, which holds the frame images,  `meta_expressions.json`, which provides referring expressions and metadata of videos, and `mask_dict.json`, which contains the ground-truth masks of objects. Ground-truth segmentation masks are saved in the format of COCO RLE, and expressions are organized similarly like Refer-Youtube-VOS.

Please note that while annotations for all frames in the **Train** set and the **Val<sup>u</sup>** set are provided, the **Val** set only provide frame images and referring expressions for inference. 

```
mevis
├── train                       // Split Train
│   ├── JPEGImages
│   │   ├── <video #1  >
│   │   ├── <video #2  >
│   │   └── <video #...>
│   │
│   ├── mask_dict.json
│   └── meta_expressions.json
│
├── valid_u                     // Split Val^u
│   ├── JPEGImages
│   │   └── <video ...>
│   │
│   ├── mask_dict.json
│   └── meta_expressions.json
│
└── valid                       // Split Val
    ├── JPEGImages
    │   └── <video ...>
    │
    └── meta_expressions.json

```

## Method Code Installation:

Please see [INSTALL.md](https://github.com/henghuiding/MeViS/blob/main/INSTALL.md)

## Inference

Obtain the output masks:
```
python train_net_lmpm.py \
    --config-file configs/lmpm_SWIN_bs8.yaml \
    --num-gpus 8 --dist-url auto --eval-only \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [output_dir]
```
Obtain the results on Val<sup>u</sup>:
```
python tools/eval_mevis.py
```
## Training

Firstly, download the backbone weights (`model_final_86143f.pkl`) and convert it using the script:

```
wget https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl
python tools/process_ckpt.py
```

Then start training:
```
python train_net_lmpm.py \
    --config-file configs/lmpm_SWIN_bs8.yaml \
    --num-gpus 8 --dist-url auto \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [path_to_weights]
```

Note: We also support training ReferFormer by providing [`ReferFormer_dataset.py`](https://github.com/henghuiding/MeViS/blob/main/ReferFormer_dataset.py)

## Models

Our results on Val<sup>u</sup> set and Val set of MeViS dataset.
* Val<sup>u</sup> set is used for offline evaluation by userself, like doing ablation study
* Val set is used for CodaLab online evaluation by MeViS dataset orgnizers
<table border="0.6">
<tbody>
    <tr>
        <th  rowspan="2" align="center" bgcolor="BBBBBB">Backbone</th>
        <th colspan="3" align="center" bgcolor="BBBBBB">Val<sup>u</sup></th>
        <th colspan="3" align="center" bgcolor="BBBBBB">Val</th>
    </tr>
    <tr>
      <td align="center" bgcolor="E5E5E5">J&F</td>
      <td align="center" bgcolor="E5E5E5">J</td>
      <td align="center" bgcolor="E5E5E5">F</td>
      <td align="center" bgcolor="E5E5E5">J&F</td>
      <td align="center" bgcolor="E5E5E5">J</td>
      <td align="center" bgcolor="E5E5E5">F</td>
    </tr>
    <tr>
      <td align="center" bgcolor="E5E5E5">Swin-Tiny & RoBERTa</td>
      <td align="center" bgcolor="E5E5E5">40.23</td>
      <td align="center" bgcolor="E5E5E5">36.51</td>
      <td align="center" bgcolor="E5E5E5">43.90</td>
      <td align="center" bgcolor="E5E5E5">37.21</td>
      <td align="center" bgcolor="E5E5E5">34.25</td>
      <td align="center" bgcolor="E5E5E5">40.17</td>
    </tr>
  </tbody>
  <colgroup>
    <col>
    <col>
    <col>
    <col>
    <col>
    <col>
    <col>
    <col>
    <col>
  </colgroup>
</table>


☁️ [Google Drive](https://drive.google.com/file/d/1djNwwNAyAIEJMZIQQHV_NYnlc8TeA4wU/view?usp=drive_link)

## Acknowledgement

This project is based on [VITA](https://github.com/sukjunhwang/VITA), [GRES](https://github.com/henghuiding/ReLA), [Mask2Former](https://github.com/facebookresearch/Mask2Former), and [VLT](https://github.com/henghuiding/Vision-Language-Transformer). Many thanks to the authors for their great works!

## BibTeX
Please consider to cite MeViS if it helps your research.

```latex
@inproceedings{MeViS,
  title={{MeViS}: A Large-scale Benchmark for Video Segmentation with Motion Expressions},
  author={Ding, Henghui and Liu, Chang and He, Shuting and Jiang, Xudong and Loy, Chen Change},
  booktitle={ICCV},
  year={2023}
}
```

```latex
@inproceedings{GRES,
  title={{GRES}: Generalized Referring Expression Segmentation},
  author={Liu, Chang and Ding, Henghui and Jiang, Xudong},
  booktitle={CVPR},
  year={2023}
}
```

```latex
@article{VLT,
  title={{VLT}: Vision-language transformer and query generation for referring segmentation},
  author={Ding, Henghui and Liu, Chang and Wang, Suchen and Jiang, Xudong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```
    
A majority of videos in MeViS are from [MOSE: Complex Video Object Segmentation Dataset](https://henghuiding.github.io/MOSE/).
```latex
@inproceedings{MOSE,
  title={{MOSE}: A New Dataset for Video Object Segmentation in Complex Scenes},
  author={Ding, Henghui and Liu, Chang and He, Shuting and Jiang, Xudong and Torr, Philip HS and Bai, Song},
  booktitle={ICCV},
  year={2023}
}
```
    
MeViS is licensed under a CC BY-NC-SA 4.0 License. The data of MeViS is released for non-commercial research purpose only.
