#!/bin/bash

# docker run -it --entrypoint="" \
#     -v /Users/tim/Downloads/agespan_7T:/data \
#     -v /Users/tim/Desktop/test:/output \
#     pennsive/qsm:latest bash -c \
#     "dcm2niix -o /output -f magnitude /data/004/7T_6months_2019-10-29/memp2rage_INV2_mag_dicoms && \
#     dcm2niix -o /output -f phase /data/004/7T_6months_2019-10-29/memp2rage_INV2_phase_dicoms && \
#     /bet2/bin/bet2 /output/magnitude_e1.nii /output/magnitude_bet2 && \
#     tgv_qsm -p /output/phase_e1_ph.nii -m /output/magnitude_bet2_mask.nii.gz -f 7.0 -t 0.0019 -s -o qsm"

# docker run -it --entrypoint=bash \
#     -v /Users/tim/Downloads/agespan_7T:/data \
#     -v /Users/tim/Desktop/test2:/output \
#     pennsive/qsm:latest bash -c \
#     "/bet2/bin/bet2 /data/7T-agespanT2s/MEGRE/S009/S009/6mo/reorient-N4-megre_e1.nii.gz /output/magnitude_bet2 && \
#     tgv_qsm -p /output/magnitude_bet2.nii.gz -m /output/magnitude_bet2_mask.nii.gz -f 7.0 -t 0.006 -s -o qsm"


docker run -it \
    -v /Users/tim/Downloads/agespan_7T:/data \
    -v /Users/tim/Desktop/test:/output \
    pennsive/qsm:latest --phase-dir /data/004/7T_6months_2019-10-29/memp2rage_INV2_phase_dicoms --mask-dir /data/004/7T_6months_2019-10-29/memp2rage_INV2_mag_dicoms -s