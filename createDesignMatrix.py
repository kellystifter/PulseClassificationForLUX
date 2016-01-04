import scipy as sp
import pandas as pd
import imp

#load the RQ reader
RQReader_PyMod = imp.load_source('RQReader_PyMod','LUXcode/Trunk/DataProcessing/PythonModules/Utilities/RQReader_PyMod.py')

#read in the user classifications
classification = pd.read_csv('classifications.csv')

#list the features to pull
features = ['aft_t0_samples','aft_t05_samples','aft_t25_samples','aft_t1_samples',
            'aft_t75_samples','aft_t95_samples','aft_t2_samples','pulse_std_phe_per_sample',
            'hft_t0_samples','hft_t10l_samples','hft_t50l_samples','hft_t1_samples','hft_t50r_samples',
            'hft_t10r_samples','hft_t2_samples','pre_pulse_area_positive_phe',
            'pre_pulse_area_negative_phe','post_pulse_area_positive_phe','post_pulse_area_negative_phe']
features += ['amis1_fraction','skinny_pulse_start_samples','skinny_pulse_end_samples','pulse_area_phe',
             'skinny_pulse_area_phe','pulse_area_positive_phe','pulse_area_negative_phe',
             'pulse_height_phe_per_sample','prompt_fraction','prompt_fraction_tlx',
             'top_bottom_ratio','top_bottom_asymmetry','exp_fit_amplitude_phe_per_sample',
             'exp_fit_tau_fall_samples','exp_fit_time_offset_samples']
features += ['exp_fit_tau_rise_samples','exp_fit_chisq','exp_fit_dof',
             'gaus_fit_amplitude_phe_per_sample','gaus_fit_mu_samples',
             'gaus_fit_sigma_samples','gaus_fit_chisq','gaus_fit_dof',
             's2filter_max_area_diff','s2filter_max_s2_area','s2filter_max_s1_area',
             'rms_width_samples','pulse_length_samples','pulse_classification']

#create dataframe with columns from features
designMatrix = pd.DataFrame(index=classification.index, columns=features)
#pull the user classification into the dataframe
designMatrix['user_classification'] = classification['user_classification']

#for each classification, go and pull the information from each feature using the RQ reader
for i in classification.index:
  if i%100 == 0:
    print("Working on entry %d/%d" % (i,len(classification.index)))
  filename_prefix = classification.loc[i,'filename_prefix']
  filenumber = classification.loc[i,'filenumber']
  event_number = classification.loc[i,'event_number']
  pulse_start_samples = classification.loc[i,'pulse_start_samples']
  user_classification = classification.loc[i,'user_classification']

  if ( i < 298 ):
    filename_prefix = filename_prefix[:5] + "_" + filename_prefix[5:]

  if ( i < 484 ):
    cp = "cp10506"
  else:
    cp = "cp11248"

  filename = "../../../../eliza3/lux/rq/" + filename_prefix + "_" + cp + "/" + filename_prefix + "_" + filenumber + "_" + cp + ".rq.gz"
  rqreader = RQReader_PyMod.ReadRQFile(filename,silent=True)
  pulse_number = list(rqreader[0]['pulse_start_samples'][event_number]).index(pulse_start_samples)

  for feature in features:
    designMatrix.loc[i,feature] = rqreader[0][feature][event_number][pulse_number]

#create a csv file from the dataframe
designMatrix.to_csv('designMatrix', index=False)
