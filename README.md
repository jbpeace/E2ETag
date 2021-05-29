# E2ETag
E2ETag: An End-to-End Trainable Method for Generating and Detecting Fiducial Markers<br/>
Link to our paper: https://www.bmvc2020-conference.com/assets/papers/0890.pdf <br/>

## Description
E2ETag proposes an end-to-end trainable method for designing, detecting, and enabling fiducial markers with deep learning.  Our method is made possible by introducing back-propagatable marker augmentation and superimposition into training.  The detector used was a modified DeepLabV3+ encoder which predicts marker's localization, projective pose, and class.  The images used for superimposition training were from the ImageNet dataset.  Results demonstrate that our method outperforms existing fiducial markers in ideal conditions and especially in the presence of motion blur, contrast fluctuations, noise, and off-axis viewing angles. <br/>

## Trained Markers (30 Classes)
<img src='tags/marker01.png' /> <img src='tags/marker02.png' /> <img src='tags/marker03.png' /> <img src='tags/marker04.png' /> <img src='tags/marker05.png' /> </br>
<img src='tags/marker06.png' /> <img src='tags/marker07.png' /> <img src='tags/marker08.png' /> <img src='tags/marker09.png' /> <img src='tags/marker10.png' /> </br>
<img src='tags/marker11.png' /> <img src='tags/marker12.png' /> <img src='tags/marker13.png' /> <img src='tags/marker14.png' /> <img src='tags/marker15.png' /> </br>
<img src='tags/marker16.png' /> <img src='tags/marker17.png' /> <img src='tags/marker18.png' /> <img src='tags/marker19.png' /> <img src='tags/marker20.png' /> </br>
<img src='tags/marker21.png' /> <img src='tags/marker22.png' /> <img src='tags/marker23.png' /> <img src='tags/marker24.png' /> <img src='tags/marker25.png' /> </br>
<img src='tags/marker26.png' /> <img src='tags/marker27.png' /> <img src='tags/marker28.png' /> <img src='tags/marker29.png' /> <img src='tags/marker30.png' /> </br>

## Sample Detection Visualizations
<img src='output/images/00001.png' /> </br>

<img src='output/images/00045.png' /> </br>

<img src='output/images/00133.png' />

## Citation
If you find our work useful in your research, please consider citing:

	@article{peace2020e2etag,
	  title={E2ETag: An End-to-End Trainable Method for Generating and Detecting Fiducial Markers},
	  author={J. Brennan Peace and Eric Psota and Yanfeng Liu and Lance C. Pérez},
	  journal={The 31st British Machine Vision Conference (BMVC)},
	  year={2020}
	}
