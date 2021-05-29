# E2ETag
E2ETag: An End-to-End Trainable Method for Generating and Detecting Fiducial Markers<br/>
Link to our paper: https://www.bmvc2020-conference.com/assets/papers/0890.pdf <br/>

E2ETag proposes an end-to-end trainable method for designing, detecting, and enabling fiducial markers with deep learning.  Our method is made possible by introducing back-propagatable marker augmentation and superimposition into training.  The detector used was a modified DeepLabV3+ encoder which predicts marker's localization, pose, and class.  The images used for superimposition training were from the ImageNet dataset.  Results demonstrate that our method outperforms existing fiducial markers in ideal conditions and especially in the presence of motion blur, contrast fluctuations, noise, and off-axis viewing angles. <br/>

## Citation
If you find our work useful in your research, please consider citing:

  @article{peace2020e2etag,
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title={E2ETag: An End-to-End Trainable Method for Generating and Detecting Fiducial Markers},
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author={J. Brennan Peace and Eric Psota and Yanfeng Liu and Lance C. Pérez},
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;journal={The 31st British Machine Vision Conference (BMVC)},
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year={2020}
  }


Citation: <br/>
