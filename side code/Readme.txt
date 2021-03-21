

1) ck_signal_nh.m
Generate the reference signals for CCA

2) fb_stcca_th_dataset_2021.m or fb_stcca_beta_dataset_2021.m
Evaluate the accuracy of the proposed algorithm (ms-etrca+ms_ecca) with or without subject transfer using different data lengths (e.g., 0.2, 0.3, ..., 1.0) and different source subject datasets

We may run it with different settings: i) dataset_no=1; or dataset_no=2;  ii) num_of_trials=1; num_of_trials=3; (in beta dataset) or num_of_trials=5; (in tsinghua dataset). The other parameters are fixed. 

3) source_subject_template_20210214.m or source_subject_template_beta_20210214.m

Compute the SSVEP templates from the Tsinghua dataset or BETA dataset, and save them into th_subj_ssvep_template_9.mat or beta_subj_ssvep_template_9.mat.

In the file fb_stcca_th_dataset_2021.m (or fb_stcca_beta_dataset_2021.m), the above .mat files are required.


