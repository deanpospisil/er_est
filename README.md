# er_est
estimators of correlation corrected for the downward bias of noise

This package includes the estimators detailed in (Pospisil and Bair, 2020; Pospisil and Bair, 2021). Principally these estimators were designed to estimate the Pearson’s correlation coefficient squared but correcting for the downward bias induced by trial-to-trial variability.  This includes the two novel estimators for the case of measuring the correlation between a set of fixed predictions and noisy estimates being predicted and the correlation between two sets of noisy estimates. Included are methods to generate confidence intervals for these estimators. In addition are reproductions of a variety of prior estimators meant to also account for trial-to-trial variability. Runing example.py will generate simulated data then compare these estimators to each other in addition to different methods of generating confidence intervals. 
