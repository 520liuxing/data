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
session_dict['num_of_prime_spkrs'] = 2  # Should be fixed to 2
session_dict['num_of_all_spkrs'] = 5
prob_spk =(1 - 0.2)/float(session_dict['num_of_all_spkrs'])

# Probability for the states of [Silence, Overlap, speaker 1, speaker 2, speaker 3, ..., speaker N]
session_dict['dialogue_prob'] = [0.1, 0.1] + [prob_spk] * session_dict['num_of_all_spkrs']

# How many turns do you want? (-1: as many as possible)
session_dict['number_of_spk_turns'] = -1

# Random range for speaker distance for prime speaker 
session_dict['dist_prob_range_prime_spk'] = [2, 20]

# Random range for speaker distance 
session_dict['dist_prob_range_bgr_spk'] = [2, 20]

# Noise toggle
session_dict['noise'] = True

# Set the Noise range 
session_dict['noise_gain_dB_range'] = [-20, -5]

# Set randomization range for absorption coefficient. Generally [0, 0.4] is recommended.
session_dict['absorption_range'] = [0.05, 0.5]

# How many sessions do you want? (if -1: as many as possible, -2: specify the start and end session index)
session_dict['number_of_sess'] = -2

# Specify the session numbers: start and end
session_dict['start'] = 1
session_dict['end'] = 50
   
# Specify the output file tag.
session_dict['file_id'] = 'session_'

# Create the sessions.
session_creator_wrapper(**session_dict)

