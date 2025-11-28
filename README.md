# MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.7%20|%203.8%20|%203.9-blue.svg?style=&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/FudanCVL/MeViSv2)

**[🏠[Project page]](https://henghuiding.github.io/MeViS/)**&emsp; **[📄[arXiv]](https://arxiv.org/abs/2308.08544)** &emsp; **[💾[Evaluation Server v1 (legacy)]](https://www.codabench.org/competitions/11420/)**&emsp; **[🔥[Evaluation Server v2]](https://www.codabench.org/competitions/11420/)**

This repository contains code for **ICCV2023** and **TPAMI 2025** paper:

> [MeViS: A Multi-Modal Dataset for Referring Motion Expression Video Segmentation](https://ieeexplore.ieee.org/abstract/document/11130435)  
> Henghui Ding, Chang Liu, Shuting He, Kaining Ying, Xudong Jiang, Chen Change Loy, Yu-Gang Jiang
> TPAMI 2025

> [MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions](https://arxiv.org/abs/2308.08544)  
> Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Chen Change Loy  
> ICCV 2023

<table border=1 frame=void>
  <tr>
    <td><img src="https://github.com/henghuiding/MeViS/blob/page/GIF/bird.gif" width="245"></td>
    <td><img src="https://github.com/henghuiding/MeViS/blob/page/GIF/Cat.gif" width="245"></td>
    <td><img src="https://github.com/henghuiding/MeViS/blob/page/GIF/coin.gif" width="245"></td>
  </tr>
</table>

### Abstract

This paper proposes a large-scale multi-modal dataset for referring motion expression video segmentation, focusing on segmenting and tracking target objects in videos based on language description of objects’ motions. Existing referring video segmentation datasets often focus on salient objects and use language expressions rich in static attributes, potentially allowing the target object to be identiﬁed in a single frame. Such datasets underemphasize the role of motion in both videos and languages. To explore the feasibility of using motion expressions and motion reasoning clues for pixel-level video understanding, we introduce MeViS, a dataset containing 33,072 human-annotated motion expressions in both text and audio, covering 8,171 objects in 2,006 videos of complex scenarios. We benchmark 15 existing methods across 4 tasks supported by MeViS, including 6 referring video object segmentation (RVOS) methods, 3 audio-guided video object segmentation (AVOS) methods, 2 referring multi-object tracking (RMOT) methods, and 4 video captioning methods for the newly introduced referring motion expression generation (RMEG) task. The results demonstrate weaknesses and limitations of existing methods in addressing motion expression-guided video understanding. We further analyze the challenges and propose an approach LMPM++ for RVOS/AVOS/RMOT that achieves new state-of-the-art results. Our dataset provides a platform that facilitates the development of motion expression-guided video understanding algorithms in complex video scenes.

<div align="center">
  <img src="https://github.com/henghuiding/MeViS/blob/page/static/DemoImages/teaser.png?raw=true" width="100%" height="100%"/>
</div>

<p style="text-align:justify; text-justify:inter-ideograph;width:100%">Figure 1. Examples from <b>M</b>otion <b>e</b>xpressions <b>Vi</b>deo <b>S</b>egmentation (<b>MeViS</b>) showing the dataset’s nature and complexity. The selected target objects are masked in <font color="#FF6403">orange ▇</font>. The expressions in MeViS primarily focus on motion attributes, making it impossible to identify the target object from a single frame. For example, the ﬁrst example has three parrots with similar appearances, and the target object is identiﬁed as “<i>The bird ﬂying away</i>”. This object can only be recognized by capturing its motion throughout the video. The updated MeViS 2024 further provides motion-reasoning and no-target expressions, adds audio expressions alongside text, and provides mask and bounding box trajectory annotations.</p>

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
        <th align="center" bgcolor="BBBBBB">Multi-target</th>
        <th align="center" bgcolor="BBBBBB">No-target</th>
        <th align="center" bgcolor="BBBBBB">Audio</th>
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
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
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
      <td align="center" bgcolor="ECECEC">-</td>
      <td align="center" bgcolor="ECECEC">-</td>
      <td align="center" bgcolor="ECECEC">-</td>
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
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="right" bgcolor="E5E5E5"><b>MeViS 2023</b></td>
      <td align="center" bgcolor="E5E5E5"><b>ICCV&nbsp;2023</b></td>
      <td align="center" bgcolor="E5E5E5"><b>2,006</b></td>
      <td align="center" bgcolor="E5E5E5"><b>8,171</b></td>
      <td align="center" bgcolor="E5E5E5"><b>28,570</b></td>
      <td align="center" bgcolor="E5E5E5"><b>443k</b></td>
      <td align="center" bgcolor="E5E5E5"><b>4.28</b></td>
      <td align="center" bgcolor="E5E5E5"><b>1.59</b></td>
      <td align="center" bgcolor="E5E5E5"><b>Object(s)</b></td>
      <td align="center" bgcolor="E5E5E5">7,539</td>
      <td align="center" bgcolor="E5E5E5">-</td>
      <td align="center" bgcolor="E5E5E5">-</td>
    </tr>
    <tr>
      <td align="right"><b>MeViS 2024</b></td>
      <td align="center"><b>TPAMI</b></td>
      <td align="center"><b>2,006</b></td>
      <td align="center"><b>8,171</b></td>
      <td align="center"><b>33,072</b></td>
      <td align="center"><b>443k</b></td>
      <td align="center"><b>4.28</b></td>
      <td align="center"><b>1.58</b></td>
      <td align="center"><b>Object(s)</b></td>
      <td align="center">8,028</td>
      <td align="center">3,503</td>
      <td align="center">33,072</td>
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

## MeViS v2 Dataset


**Dataset Split**

- 2,006 videos & 33,458 sentences in total;
- **Train set:** 1662 videos & 27,502 sentences, used for training;
- **Val<sup>u</sup> set:** 50 videos & 907 sentences, ground-truth provided, used for offline self-evaluation (e.g., ablation study) during training;
- **Val set:** 140 videos & 2,523 sentences, ground-truth **not** provided, used for [**CodaLab online evaluation**](https://www.codabench.org/competitions/11420/);
- **Test set:** Will be progressively and selectively released and used for evaluation during the competition periods ([PVUW](https://pvuw.github.io/), [LSVOS](https://lsvos.github.io/));

It is suggested to report the results on **Val<sup>u</sup> set** and **Val set**.

## Online Evaluation

Please submit your results of **Val set** on

- 💯 v1 server (Closing Soon): [**CodaLab**](https://codalab.lisn.upsaclay.fr/competitions/15094)
- 💯 v2 server: [**CodaBench**](https://www.codabench.org/competitions/11420/).

It is strongly suggested to first evaluate your model locally using the **Val<sup>u</sup>** set before submitting your results of the **Val** to the online evaluation system.

## File Structure

The dataset follows a similar structure as [Refer-YouTube-VOS](https://youtube-vos.org/dataset/rvos/). Each split of the dataset consists of three parts: `JPEGImages`, which holds the frame images, `meta_expressions.json`, which provides referring expressions and metadata of videos, and `mask_dict.json`, which contains the ground-truth masks of objects. Ground-truth segmentation masks are saved in the format of COCO RLE, and expressions are organized similarly like Refer-Youtube-VOS.

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

###  1. Val<sup>u</sup> set
Obtain the output masks of Val<sup>u</sup> set:
```
python train_net_lmpm.py \
    --config-file configs/lmpm_SWIN_bs8.yaml \
    --num-gpus 8 --dist-url auto --eval-only \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [output_dir]
```
Obtain the J&F results on Val<sup>u</sup> set:
```
python tools/eval_mevis.py
```
###  2. Val set
Obtain the output masks of Val set for [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/15094) online evaluation:
```
python train_net_lmpm.py \
    --config-file configs/lmpm_SWIN_bs8.yaml \
    --num-gpus 8 --dist-url auto --eval-only \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [output_dir] DATASETS.TEST '("mevis_test",)'
```
### CodaLab Evaluation Submission Guideline

The submission format should be a **.zip** file containing the predicted .PNG results of the **Val set** (for current competition stage).

You can use following command to prepare .zip submission file
```
cd [output_dir]
zip -r ../xxx.zip *
```
A submission example named *sample_submission_valid.zip* can be found from the [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/15094).
```
sample_submission_valid.zip       // .zip file, which directly packages 140 val video folders
├── 0ab4afe7fb46                  // video folder name
│   ├── 0                         // expression_id folder name
│   │   ├── 00000.png             // .png files
│   │   ├── 00001.png
│   │   └── ....
│   │
│   ├── 1
│   │   └── 00000.png
│   │   └── ....
│   │
│   └── ....
│ 
├── 0fea0cb75a25
│   ├── 0                              
│   │   ├── 00000.png
│   │   └── ....
│   │
│   └── ....
│
└── ....                      
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
* Val set is used for CodaLab online evaluation by MeViS dataset organizers
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
