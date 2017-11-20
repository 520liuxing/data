from data_creation_module import *
    
session_dict = {}

# Provide your actual directory for LibriSpeech and QUT-NOISE
session_dict['librispeech_directory'] = 'LibriSpeech'
session_dict['noise_data_directory'] = 'QUT-NOISE/QUT-NOISE'
session_dict['wav_output_directory'] = './USCDiarLibri_2_6'

if_not_create(session_dict['wav_output_directory'])

# Verbose
session_dict['verbose'] = 1

# Session dialogue Settings 
session_dict['num_of_prime_spkrs'] = 2  # Should be fixed to 2. 
session_dict['num_of_all_spkrs'] = 4
prob_spk =(1 - 0.2)/float(session_dict['num_of_all_spkrs'])

session_dict['dialogue_prob'] = [0.1, 0.1] + [prob_spk] * session_dict['num_of_all_spkrs']

# How many turns do you want? (-1: as many as possible)
session_dict['number_of_spk_turns'] = -1

# Random range for speaker distance 
session_dict['dist_prob_range_bgr_spk'] = [2, 20]

# Random range for speaker distance for prime speaker 
session_dict['dist_prob_range_prime_spk'] = [2, 20]

# Noise toggle
session_dict['noise'] = False

# Set the Noise range 
session_dict['noise_gain_dB_range'] = [-20, -5]

# Set randomization range for absorption coefficient. Generally [0, 0.4] is recommended.
session_dict['absorption_range'] = [0.25, 0.25]

# How many sessions do you want? (if -1: as many as possible, -2: specify the start and end session index)
session_dict['number_of_sess'] = -2

# If 'number_of_sess' is -2, specify the session numbers: start and end
session_dict['start'] = 1
session_dict['end'] = 50
   
# Specify the output file tag.
session_dict['file_id'] = 'session_'

# Create the sessions.
session_creator_wrapper(**session_dict)




### For the session with fixed length interfering speakers ###
varying_fixed_length = [2, 4, 8, 16]

# Random range for speaker distance for prime speaker 
session_dict['dist_prob_range_prime_spk'] = [5, 5]

# Specify the session numbers: start and end
session_dict['start'] = 1
session_dict['end'] = 20

for fix_len in varying_fixed_length:
    
    # Random range for speaker distance 
    session_dict['dist_prob_range_bgr_spk'] = [fix_len, fix_len]

    # Specity the output file tag.
    session_dict['file_id'] = 'session_'+ 'fixlen_' + str(fix_len) +'L_'
    
    # Create the sessions.
    session_creator_wrapper(**session_dict)


