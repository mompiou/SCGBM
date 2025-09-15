# Shear Coupling Grain Boundary Migration scripts

## Misorientation maps

```misorientation_maps.py``` computes minimum misorientation angles from two orientation maps of the same area and highlight area where grain boudaries have moved. 

- Input:  ```angles.txt``` contains the two sets of Euler angles  ```phi_1a, phi_a,phi_2a,phi_1b, phi_b,phi_2b``` (flattened image)

- Outputs: ```rotation.txt```: the misorientation axes and angles, ```color.txt```: the associated grey levels according to the misorientation angles.


## Frank Bilby coupling factors

```beta-FB.py``` computes the coupling factor according to the Frank-Bilby equation for a given rotation axis for tilt grain boundary (cubic crystal).


## Automatic image correction during creep

Use to provide automatic geometric transformation of AFM and EBSD maps before and after creep using fiducial markers (surface imperfections).

- A deep learning model (ResUNet network, Keras implementation) was trained to segment markers . ```res_u_net-markers.py```: to train images and masks and ```res_u_net-markers-predict.py``` for inference.

- Markers centroids from segmentation maps were computed using watershed (OpenCV implementation) and geometric transformation was estimated between markers before and after creep. Need more than two markers (not the case for the images avant.bmp and apres.bmp): ```image_correction.py``` (input: two images before and after creep, output: the transformation matrix; the corrected image)


## Contributions
Romain Gautier, Frédéric Mompiou


