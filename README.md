# USCDiarLibri #

## USCDiarLibri ##
We created the USCDiarLibri dataset that can be used to test speaker diarization tasks with various customized setups and randomization. USCLibriDiar dataset is based on artificial multi-party dialogs made from noisy, reverberated audio from the LibriSpeech database and it’s highly parameterized to allow for diverse conditions. 

USCDiarLibri generates USCDiarLibri dataset using external speech corpora and noise dataset. Therefore, LibriSpeech data set and QUT-NOISE dataset should be downloaded to a certain folder before you start the data generation script.

## Download and Installation ##

### Data Preparation ###

(1) Download the following speech dataset:

   - [LibriSpeech train-clean-100 dataset](http://www.openslr.org/resources/12/train-clean-100.tar.gz)

   - Further information about this speech dataset can be found at: [LibriSpeech Information](http://www.openslr.org/11/)


(2) Download the following noise dataset:

   - [QUT-NOISE Dataset](https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/8342a090-89e7-4402-961e-1851da11e1aa/download/qutnoise.zip)


(3) The directory which includes USCDiarLibri should be setup as the following.

     SCUBA-USCDiarLibri
      +--QUT-NOISE         
      |   +--QUT-NOISE-TIMIT/        
      |   +--QUT-NOISE-NIST2008/     
      |   +--QUT-NOISE/
      |   +--docs/
      |   +--code/
	  +--LibriSpeech         
      |   +--train-clean-100/ 
      |   +--BOOKS.TXT
      |   +--CHAPTERS.TXT
      |   ...
      +--train-clean-100-json
      |   +--103/
      |   +--1034/
      |   +--1040/
      |   ...
      +--Libre_file_list.txt   
      +--QUT_noise_list.txt
      +--README.md
      +--data_creation_module.py
      +--USCDiarLibri_2_4.py
      +--USCDiarLibri_2_6.py
      +--USCDiarLibri_gen.py
      
      
## Prerequisites ##
- [Python3.6+](https://www.python.org/downloads/)
- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/install.html)
- [librosa](https://github.com/librosa/librosa)
- [soundfile](https://pypi.python.org/pypi/SoundFile/0.8.1)
- [glob](https://pypi.python.org/pypi/glob2)
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)

## Creating USCLibriDiar Dataset ##

- For pre-setup dataset, run the given python script.
```
$python USCDiarLibri_2_4.py
$python USCDiarLibri_2_6.py
```
- For customizable dataset, modify the parameters in USCDiarLibri_gen.py. The parameters in USCDiarLibri_gen.py are defined in the form of python dictionary as below:
```
session_dict['parameter_name'] = [Value]
```
- For the parameter descriptions, read the following descriptions. 

## Parameters and Descriptions ##

The following descriptions are for parameters of USCDiarLibri_gen.py. The randomization is done session by session.
> **librispeech_directory**: String. The directory path for Downloaded LibriSpeech data.

> **noise_data_directory**: String. The directory path for Downloaded QUT-NOISE data.

> **wav_output_directory**: String. The directory path for generated .wav files.

> **verbose**: Python Boolean: True or False. Display messages along the data generation process.

> **num_of_prime_spkrs**: Positive integer. This parameter determines the number of primary speakers. Currently, the number of primary speakers is fixed to 2.

> **num_of_all_spkrs**: Positive integer. The number of total speakers per a session. This number includes both primary speakers and interfering speakers.

> **dialogue_prob**: Python list: probablility for the states of [Silence, Overlap, speaker 1, speaker 2, speaker 3, ..., speaker N]. If you set bigger probability to a certain state than others, the state will appear more frequently than other states.

> **number_of_spk_turns**: Positive integer or -1. The number of speaker turns in a session. Put -1 if you want to create as many turns as possible. A turn means a change of state in artificial dialogue. For example, if there are three turns in a session, the example session could be speech signal of Speaker1 for 2.3sec followed by silence for 1.8sec followed by speech signal of speaker5 for 3.6sec.  

> **dist_prob_range_prime_spk**: Python list: [Min, Max]. Determines the range of uniform random variable for distance between two primary speakers.

> **dist_prob_range_bgr_spk**: Python list: [Min, Max]. Determines the range of uniform random variable for distance between microphone and interfering speakers.

> **noise**: Python Boolean: True or False. Toggle the background noise.

> **noise_gain_dB_range**: Python list: [Min, Max]. Determines the range of uniform random variable for the Signal to Noise Ratio (SNR) in dB scale.

> **absorption range**: Python list: [Min, Max]. Determines the range of uniform random variable for the absorption coefficient of virtual room that simulates impulse response. If you put 0, you get unechoic signal. 

> **number_of_sess**: Positive integer, -1 or -2. The number of sessions you want to create. If you put -1, the system generates maximum number of sessions. If you If you want to create the specific interval of sessions, use option of -2 and specify minimum and maximum index number.

> **start**: Positive integer. Minimum index number.

> **end**: Positive integer. Maximum index number. 

> **file_id**: String. Determines the tag for the name of the output file.

## Generated Dataset ##

USCDiarLibri script generates three different kind of files. 

- WAV file - session_[N]_ch[M].wav : Wav file contains output from each microphone. It contains speech signal from primary speakers, interfering speakers and noise. 

- JSON file - session_[N]_ch[M].json : json file that contains word alignment information for each channel. the information includes alignedword, start and end time, duration of each phoneme, and ending time. 

- RTTM file - session_[N].rttm : RTTM format is an evaluation format for NIST RichTranscription dataset. Please refer to [The Rich Transcription 2006 Spring Meeting Recognition Evaluation](https://link.springer.com/chapter/10.1007/11965152_28) 
