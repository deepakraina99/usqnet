# USQNet

This is the official source code repository for the paper titled "Deep Learning Model for Quality Assessment of Urinary Bladder Ultrasound Images using Multi-scale and Higher-order Processing" (Paper link). The focus of our work is to address the challenges in Autonomous Ultrasound Image Quality Assessment (US-IQA). 

### Installations
````
conda env create -f environment.yml
````

### Training
````
python train.py --data_dir dataset/{dir_name} --num_epochs --folds --lr --batch_size
````
The trained model will be saved in the outputs folder.

### Testing
````
python test.py --data_dir dataset/{dir_name} --load_model outputs/{model_name}.pth
````

### Citation
```
@article{raina2024deep,
  title={Deep Learning Model for Quality Assessment of Urinary Bladder Ultrasound Images using Multi-scale and Higher-order Processing},
  author={Raina, Deepak and SH, Chandrashekhara and Voyles, Richard and Wachs, Juan and Saha, Subir Kumar},
  journal={IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control},
  volume={??},
  number={?},
  pages={???--???},
  year={2023},
  publisher={IEEE}
}
```
### License
[![Creative Commons License](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)  
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/) for Noncommercial use only. Any commercial use should obtain formal permission.

### Acknowledgement
This code base is built upon [ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py), [Res2Net](https://github.com/Res2Net/Res2Net-PretrainedModels) and [MPN-COV](https://github.com/jiangtaoxie/MPN-COV). Thanks to the authors of these papers for making their code available for public usage.  
