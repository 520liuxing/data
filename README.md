# USCDiarLibri #

## USCDiarLibri ##
We created the USCDiarLibri dataset that can be used to test speaker diarization tasks with various customized setups and randomization. USCLibriDiar dataset is based on artificial multi-party dialogs made from noisy, reverberated audio from the LibriSpeech database and it’s highly parameterized to allow for diverse conditions. 

USCDiarLibri generates USCDiarLibri dataset using external speech corpora and noise dataset. 

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
- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/install.html)
- [librosa](https://github.com/librosa/librosa)
- [soundfile](https://pypi.python.org/pypi/SoundFile/0.8.1)
- [glob](https://pypi.python.org/pypi/glob2)
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)

## Creating USCLibriDiar Dataset ##

- For pre-setup dataset, run the given python script.
```
$python USCDiarLibri_2_4
$python USCDiarLibri_2_6
```
- For customizable dataset, modify the parameters in USCDiarLibri_gen

## Parameters and Descriptions ##

The following descriptions are for parameters of USCDiarLibri_2_X.py

- num_of_prime_speakers: This parameter determines the number of primary speakers. Currently, the number of primary speakers is fixed to 2.
- num_of_all_speakers: The number of total speakers per a session. This number includes both primary speakers and interfering speakers.
- number_of_sess: The number of sessions you want to create.
- number_of_spk_turns: The number of speaker turns in a session. Put -1 if you want to create as many turns as possible.
- dist_prob_range: Determines the range of uniform random variable for distance between microphone and interfering speakers.
- dist_prob_range_prime_spk: Determines the range of uniform random variable for distance between two primary speakers.




