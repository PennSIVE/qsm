QSM reconstruction using Total Generalized Variation (TGV-QSM)
Kristian Bredies and Christian Langkammer
Version from June 2016
www.neuroimaging.at

=== NEW ===
- It is a python package now which simplifies installation
- Orientation fixed: image data does NOT have to be transversal any longer (thx Daniel!)


=== Installation ===
python setup.py install
 (in case of problems such as Ubuntu 18.04+ try installing an older version of cython with pip install Cython==0.19.2 - Thanks Andrej!) 

=== Command line options ===

usage: tgv_qsm [-h] -p PHASE -m MASK [-o OUTPUT_SUFFIX]
               [--alpha ALPHA ALPHA | --factors FACTORS [FACTORS ...]]
               [-e EROSIONS] [-i ITERATIONS [ITERATIONS ...]]
               [-f FIELDSTRENGTH] [-t ECHOTIME] [-s] [--ignore-orientation]
               [--save-laplacian] [--output-physical] [--no-resampling]
               [--vis] [-v]

remarks for options:
	-t TE in seconds
	-f fieldstrength in Tesla
	-s autoscaling for SIEMENS phase data


test data:
tgv_qsm  -p epi3d_test_phase.nii.gz -m epi3d_test_mask.nii.gz -f 2.89 -t 0.027 -o epi3d_test_QSM





=== bet brain masking  ===

for high res data (~0.5 mm iso) this does quite a good job:
bet magni.nii.gz mask -n -m -R -f 0.1 -g 0.0
(in case something is lost, vary parameter g minimally)

