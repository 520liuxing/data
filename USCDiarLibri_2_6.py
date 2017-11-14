from data_creation_module import *
import sys 
import os 
    
session_dict = {}

# Provide your actual directory of TEDLIUM data and RIR data
session_dict['Noise_Data_directory'] = '/home/tango_guest/QUT-NOISE/QUT-NOISE'
session_dict['LibriSpeech_directory'] = '/home/tango_guest/LibriSpeech'

session_dict['wav_output_directory'] = './USCDiarLibri_2_6'
if_not_create(session_dict['wav_output_directory'])

# Session dialogue Settings
session_dict['SR'] = 16000
session_dict['num_of_prime_spkrs'] = 2
session_dict['num_of_all_spkrs'] = 6
prob_spk =(1 - 0.2)/float(session_dict['num_of_all_spkrs'])

session_dict['dialogue_prob'] = [0.1, 0.1] + [prob_spk] * session_dict['num_of_all_spkrs']

# data_list_mat, number_index_mat, speaker_id_list = read_TEDLIUM_testset(TEDLIUM_directory)
dir_list, number_of_sess = read_LibriSpeech_index(session_dict['LibriSpeech_directory'], 
                                                  session_dict['num_of_all_spkrs'],
                                                  session_dict['num_of_prime_spkrs'])

# Dir list
session_dict['dir_list'] = dir_list
session_dict['number_of_sess'] = number_of_sess

# How many sessions do you want? (-1: as many as possible)
session_dict['number_of_sess'] = 3

# How many speaker chnages do you want? (-1: as many as possible)
session_dict['number_of_spk_turns'] = 20

# Random range for speaker distance 
session_dict['dist_prob_range'] = [2, 20]

# Random range for speaker distance for prime speaker 
session_dict['dist_prob_range_prime_spk'] = [2, 20]

# Noise toggle
session_dict['noise'] = True

# Set the Noise range 
session_dict['noise_gain_dB_range'] = [0, 10]

# Session number: start and end
session_dict['start'] = 1
session_dict['end'] = 50
    
session_dict['file_id'] = 'session_'

session_creator_wrapper(**session_dict)

### For the session with fixed length interfering speakers ###

varying_fixed_length = [2, 4, 8, 16]

# Random range for speaker distance for prime speaker 
session_dict['dist_prob_range_prime_spk'] = [5, 5]
session_dict['start'] = 1
session_dict['end'] = 20

for fix_len in varying_fixed_length:
    
    # Random range for speaker distance 
    session_dict['dist_prob_range'] = [fix_len, fix_len]

    session_dict['file_id'] = 'session_'+ 'fixlen_' + str(fix_len) +'L_'

    session_creator_wrapper(**session_dict)

