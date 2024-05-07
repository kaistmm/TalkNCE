## TalkNCE: Improving Active Speaker Detection with Talk-Aware Contrastive Learning

Official implementation of TalkNCE. [[Paper]](https://mmai.io/pubs/pdfs/jung24a.pdf)

### TalkNCE loss

This repository contains [LoCoNet](https://arxiv.org/pdf/2301.08237) model trained with TalkNCE loss.

TalkNCE loss is implemented in ``Loconet()`` class of ``loconet.py``

Pretrained checkpoint can be downloaded [here](https://drive.google.com/file/d/1k8sS3Io6dVMFKqluvoORt6u6MaEgvQOG/view?usp=sharing).
This checkpoint is a LoCoNet model trained with TalkNCE loss on the training set of AVA dataset.
It yields mAP 95.5% on the validation set of AVA dataset.

### Dependencies

Start from building the environment
```
conda create -n {env_name} python=3.7.9
conda activate {env_name}
pip install -r requirements.txt
```


### Data preparation

We follow TalkNet's data preparation script to download and prepare the AVA dataset.

```
python train.py --dataPathAVA {AVADataPath} --download 
```

`AVADataPath` is the folder you want to save the AVA dataset and its preprocessing outputs, the details can be found in [here](https://github.com/syl4356/TalkNCE_ASD/blob/main/utils/tools.py) . Please read them carefully.

After AVA dataset is downloaded, please change the ``DATA.dataPathAVA`` entry in the config file (``configs/multi.yaml``). 

If you have already downloaded AVA-ActiveSpeaker dataset, download csv files only by running this command:

```
gdown 1h8DISV9sYHGi2CsDI7PXK4kXmS2_kjEg
unzip talknce_csv.zip
rm talknce_csv.zip
```

### Training script
```
python -W ignore::UserWarning train.py --cfg configs/multi.yaml OUTPUT_DIR {output directory}
```



### Test script

```
python -W ignore::UserWarning test_multicard.py --cfg configs/test.yaml  RESUME_PATH {pretrained ckpt path}
```




### Citation

If you find this code useful, please consider citing the paper below:
```
@inproceedings{jung2023talknce,
  title={TalkNCE: Improving Active Speaker Detection with Talk-Aware Contrastive Learning},
  author={Jung, Chaeyoung and Lee, Suyeon and Nam, Kihyun and Rho, Kyeongha and Kim, You Jin and Jang, Youngjoon and Chung, Joon Son},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
  year={2024}
}
```


### References

Our baseline code is based on the following repositories:

- [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD) 
- [LoCoNet](https://github.com/SJTUwxz/LoCoNet_ASD).


