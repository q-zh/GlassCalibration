This repo provides the code for our CVPR paper "What does Plate Glass Reveal about Camera Calibration?". \[[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_What_Does_Plate_Glass_Reveal_About_Camera_Calibration_CVPR_2020_paper.pdf)\]
\[[poster](https://1drv.ms/b/s!AqqMkGs8p4aNgi68AvUiC_mBttni?e=7OdPQD)\]
\[[video](https://1drv.ms/v/s!AqqMkGs8p4aNgi-Is7Ah2ZgHrnfF?e=JezfFD)\]

##

The code is modified based on that in https://github.com/csqiangwen/Single-Image-Reflection-Removal-Beyond-Linearity with a similar environment setup. 

##
Please find the pre-trained model in https://pan.baidu.com/s/17OeeQTfhZUR4zGjG6eNDgw  code: chjq

Please find the WILD testing data in https://pan.baidu.com/s/1qp36IdUeM8K17-Wa9ppQXA code: cvpr  and CONTROLLED testing data in https://pan.baidu.com/s/13oQGoP5G8RKdlj9CGHD9mA code：cvpr 

The numbers in the ground truth file represent the focal length (mm) and orientation of glass.


Original raw data can be find here https://pan.baidu.com/s/1iRw0hLDDlmrASh5e9o2Tfg code：cvpr 

## 
Once got the ouput of a 3-dimensional vector (x,y,z), the FoV can be calculated by atan(0.5 / z).

##
If you find our code is useful, please cite our paper. If you have any problem of implementation or running the code, please contact us: csqianzheng@gmail.com
```
@inproceedings{zheng2020does,
  title={What Does Plate Glass Reveal About Camera Calibration?},
  author={Zheng, Qian and Chen, Jinnan and Lu, Zhan and Shi, Boxin and Jiang, Xudong and Yap, Kim-Hui and Duan, Ling-Yu and Kot, Alex C},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3022--3032},
  year={2020}
}
```
