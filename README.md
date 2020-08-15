# Density Perception (Master's research project)
<ul>
  <li> Implemented applications with PsychoPy to produce visual stimuli for performing visual perception experiments and collecting data to build computational models </li>
  <li> Simulated humans’ estimate of density/number of objects using Spatial Frequency(SF) filters and Image Occupancy models </li>
</ul>


<hr>
<h3> Sample Gaussian blobs stimulus patches </h3>
<h4> Varying the patch size affects our perception of density. Patch B has the same physical density as patch A, but the larger patch size of B makes it appear considerably denser. In contrast, although patch C has lower physical density than patch A, it perceptually matches the density of A.  </h4>
<img src="./images/Gaussian blobs stimulus patches.png">


<hr>
<h3> Example stimuli, high-SF(Hi-freq) and low-SF(Lo-freq) versions of rectified versions of the stimuli </h3>
<h4> Convolving with Laplacian of Gaussian(LOG) of different spatial frequency produced different responses. Small filters generated isolated responses to individual elements while large filters responded to clusters of elements and their responses are limited by the stimulus patch size. </h4> 
<img src="./images/high-SF and low-SF versions of stimuli.png">


<hr>
<h3> Screenshots of visual stimuli when objects are moving and occlusion between objects occurs </h3>
<img src="./images/layers of dots with occlusion.png">
