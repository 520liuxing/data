# USCDiarLibri #

## USCDiarLibri ##
We created the USCDiarLibri dataset that can be used to test speaker diarization tasks with various customized setups and randomization. USCLibriDiar dataset is based on artificial multi-party dialogs made from noisy, reverberated audio from the LibriSpeech database and it’s highly parameterized to allow for diverse conditions. 

USCDiarLibri generates USCDiarLibri dataset using external speech corpora and noise dataset. 

## Download and Installation ##

### Data Preparation ###

1. Download the following speech dataset:

   * [LibriSpeech train-clean-100 dataset](http://www.openslr.org/resources/12/train-clean-100.tar.gz)

   * Further information about this speech dataset can be found at: [LibriSpeech Information](http://www.openslr.org/11/)


2. Download the following noise dataset:

   * [QUT-NOISE Dataset](https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/8342a090-89e7-4402-961e-1851da11e1aa/download/qutnoise.zip)


3. The directory which includes USCDiarLibri should be setup as the following.

     SCUBA-USCDiarLibri
      +--QUT-NOISE         
      |   +--QUT-NOISE-TIMIT        
      |   +--QUT-NOISE-NIST2008     
      |   +--QUT-NOISE
      |   +--docs
      |   +--code
	  +--LibriSpeech         
      |   +--train-clean-100 
      |   +--sample      
      +--Libre_file_list.txt   
      +--QUT_noise_list.txt    
      
      
## Prerequisites ##
* numpy
* librosa
* soundfile
* glob
* scipy.signal
* pyroomacoustics

## Creating USCLibriDiar Dataset ##

* For pre-setup dataset, run the given python script.
```
python USCDiarLibri_2_4
```
* For customizable dataset, modify the parameters in USCDiarLibri_gen

## Parameters ##




