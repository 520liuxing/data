from __future__ import print_function
import numpy as np
import librosa
import os
import soundfile as sf
import glob
import scipy.signal
import json
import sys
import pyroomacoustics as pra
import time
from scipy import signal
try:
    import cPickle as pickle
except:
    import pickle

class session:
    def __init__(self, **kwargs):
        # Specifications of the session
        self.LibriSpeech_directory = kwargs.get('LibriSpeech_directory')
        self.Noise_Data_directory = kwargs.get('Noise_Data_directory')
        self.dir_list = kwargs.get('dir_list')
        self.ses_idx = kwargs.get('ses_idx')
        self.SR = kwargs.get('SR', 16000)
        self.random_seed = kwargs.get('random_seed', 0)
        self.dialogue_prob = kwargs.get('dialogue_prob')
        self.spk_id_sequence = kwargs.get('spk_id_sequence')
        self.ch_atten_mix = kwargs.get('ch_atten_mix')
        self.verbose = kwargs.get('verbose', 1)
        
        # Data of the session
        self.dist_prob_range_bgr_spk = kwargs.get('dist_prob_range_bgr_spk')
        self.dist_prob_range_prime_spk = kwargs.get('dist_prob_range_prime_spk')
        self.distance_prime_spk = kwargs.get('distance_prime_spk')
        self.distance_bgr_ch = kwargs.get('distance_bgr_ch')
        self.rir_spk = kwargs.get('rir_spk')
        self.rir_bgr_ch = kwargs.get('rir_bgr_ch')
        self.data_list_mat = kwargs.get('data_list_mat')
        self.num_of_all_spkrs = kwargs.get('num_of_all_spkrs', 4)
        self.num_of_prime_spkrs = kwargs.get('num_of_prime_spkrs', 2)
        self.num_of_bkg_spkrs = kwargs.get('num_of_bkg_spkrs', 2)
        self.number_of_spk_turns = kwargs.get('number_of_spk_turns', -1)
        self.number_of_spk_load = kwargs.get('number_of_spk_load', -1)

        # Randomness of the session
        self.absorption_applied = kwargs.get('absorption_applied', 0.25)
        self.absorption_range = kwargs.get('absorption_range', [0, 0.5])
        self.min_max_silence_length_sec = kwargs.get('min_max_silence_length_sec')
        self.noise = kwargs.get('noise', False)
        self.noise_ts = kwargs.get('noise_ts')
        self.noise_path = kwargs.get('noise_path')
        self.noise_gain_dB_range = kwargs.get('noise_gain_dB_range')
        self.noise_gain_dB_applied = kwargs.get('noise_gain_dB_applied')
        
        # IR related parameter
        self.IR_sample_margin = kwargs.get('IR_sample_margin')
        self.IR_cut_thrs = kwargs.get('IR_cut_thrs')

def sprint(verbose, text, val):
    if verbose:
        print(text, str(val))


def rprint(verbose, text, val):
    if verbose:
       sys.stdout.flush()
       sys.stdout.write(text+ str(val))
       sys.stdout.flush()


def if_not_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def open_and_list(stm_txt):
    with open(stm_txt) as f:
        content_mat = []
        for line in f:
            content_mat.append(line)
    # Remove whitespace characters like `\n` at the end of each line.
    content_mat = [x.strip() for x in content_mat]
    return content_mat


def silence_remover(input_ts, threshold_db):
    intervals = librosa.effects.split(input_ts, top_db=threshold_db)
    output_ts = []
    for (st, en) in intervals:
        output_ts.extend(input_ts[st:en])
    return output_ts 


def random_word_length(seed):
    mean = 2.5 * (180/60)  # 160 word per min, avg speech dur 2.15
    scale = mean/np.sqrt(np.pi/2)
    return int(max(np.ceil(np.random.rayleigh(scale)),1))


def return_json_dict(json_path):
    with open(json_path) as data_handle:
        data = json.load(data_handle)
    return data


def json_validity(verbose, data, start, end):
    check_list = []
    for k in range(start, end):
        check_list.append(data['words'][k]['case'])
        # if data['words'][k]['case'] != 'success':
            # sprint(verbose, 'Defective aligning!', '')
    result = True if check_list.count('success') == len(check_list) else False
    return result


def create_rand_utt(verbose, data, seed):
    n_total_word = len(data['words'])
    word_lens_mat = []
    cum_n = 0
    while True:
        word_lens_mat.append(random_word_length(seed)) 
        if sum(word_lens_mat) > n_total_word:
            del word_lens_mat[-1]
            break
    json_div_out = []
    segment_durations = []
    cum_sum_wm = np.cumsum(word_lens_mat)
    for ct, el in enumerate(word_lens_mat):
        if ct == 0:
            if json_validity(verbose, data, 0, cum_sum_wm[ct]):
                json_div_out.append(data['words'][0:word_lens_mat[ct]])
                segment_durations.append( [data['words'][0]['start'], data['words'][cum_sum_wm[ct]-1]['end']] )
        elif ct > 0:
            if json_validity(verbose, data, cum_sum_wm[ct-1], cum_sum_wm[ct]):
                json_div_out.append(data['words'][cum_sum_wm[ct-1]:cum_sum_wm[ct]])
                segment_durations.append( [data['words'][cum_sum_wm[ct-1]]['start'], data['words'][cum_sum_wm[ct]-1]['end']] )
             
    return word_lens_mat, json_div_out, segment_durations


def get_sim_RIR(distance, width, absp_coeff):
    
    SR = 16000
    # room dimension
    y_val = np.sqrt( max( distance**2 - (width/float(4))**2, 1) )
    room_dim = [width, y_val + 3, 3]

    # Create the shoebox
    shoebox = pra.ShoeBox(
        room_dim,
        absorption=absp_coeff,
        fs=SR,
        max_order=20,
        )

    # source and mic locations
    shoebox.add_source([width*0.25, 1, 2])
    shoebox.add_microphone_array(
            pra.MicrophoneArray(
                np.array([[width*0.5, 2+y_val, 2]]).T,
                shoebox.fs)
            )

    # run ism
    shoebox.image_source_model()
    shoebox.compute_rir()
    rir = shoebox.rir[0][0]
    
    return rir, shoebox


def offset_remover(elem_ts):
    return elem_ts - np.mean(elem_ts)


def normalize_wave(in_ts):
    in_ts = np.array(in_ts)
    return in_ts / np.max(np.abs(in_ts))


def read_and_divide_file(flac_file_address, segment_durations, SR):
    buff_ts, _ = librosa.core.load(flac_file_address, sr=SR, dtype=np.float32)
    audio_lists = []
    for ct, intvl in enumerate(segment_durations):
        start = int(intvl[0] * SR)  # in sec
        end = int(intvl[1] * SR)  # in sec
        alpha = 0.05
        offset_removed = offset_remover(buff_ts[start:end])
        windowed_buff_ts = np.multiply(offset_removed, signal.tukey(len(offset_removed), alpha=alpha))
        audio_lists.append(list(windowed_buff_ts))

    return audio_lists


def random_audio_split(flac_file_address, max_utt, SR):
    max_utt = 4
    rand_num = np.random.randint(1, max_utt) + 1
    rmse_frame_size = 2048
    hop_length = rmse_frame_size / 4
    buff_ts, _ = librosa.load(flac_file_address, sr=SR)

    buff_rmse = np.transpose(librosa.feature.rmse(buff_ts,
                                                  frame_length=rmse_frame_size,
                                                  hop_length=hop_length))
    length_rmse = len(buff_rmse)
    seeking_range = length_rmse / 10

    length_rmse = len(buff_rmse)
    index_min = np.int64((length_rmse / rand_num) *
                         np.linspace(0, rand_num, rand_num, endpoint=False))

    index_list = []
    for k in range(1, rand_num):
        sort_rmse_idx = np.argsort(
            buff_rmse[(index_min[k] - seeking_range):(index_min[k] + seeking_range)].T)[::-1]
        index_list.append(index_min[k] - seeking_range + sort_rmse_idx[0][0])

    index_list.insert(0, 0)
    index_list.append(length_rmse)
    out_ts_data = []
    for k in range(len(index_list) - 1):
        start_ts = index_list[k] * hop_length
        end_ts = index_list[k + 1] * hop_length
        out_ts_data.append(buff_ts[start_ts:end_ts])


    return out_ts_data


def uni_rand(rand_range, seed):
    np.random.seed(seed)
    return np.random.uniform(rand_range[0], rand_range[1])


def uni_rand_noseed(rand_range):
    return np.random.uniform(rand_range[0], rand_range[1])


def load_random_filename(verbose, list_name, seed):
    list_path = list_name + '.txt'
    sprint(verbose, list_path, '')
    with open(list_path) as f:
        content = []
        for line in f:
            line = line.strip()
            content.append(line)
    np.random.seed(seed)
    call_idx = np.random.randint(0, len(content) - 1, 1)[0]
    return content[call_idx]


def read_LibriSpeech_index(verbose, LibriSpeech_directory, num_of_all_spkrs, num_of_prime_spkrs):
    num_of_bkg_spkrs = num_of_all_spkrs - num_of_prime_spkrs
    SR = 16000
    audio_folder_name = 'train-clean-100'
    list_path = 'Libre_file_list.txt'
    data_path = LibriSpeech_directory + '/' + audio_folder_name + '/'
    sprint(verbose, 'data_path', data_path)
    
    content = open_and_list(list_path)
    number_of_sess = int(np.floor((len(content) - num_of_bkg_spkrs) / num_of_all_spkrs ))

    dir_list = []
    for k in range(number_of_sess):
        buff = []
        for j in range(num_of_all_spkrs):
            buff.append(content[num_of_all_spkrs * k + j])
        dir_list.append(buff)
    return dir_list, number_of_sess


def add_seesion_info2json(ch_json, session):
    info_dict = {
    'session index': str(session.ses_idx),
    'dialogue_prob': str(session.dialogue_prob), 
    'spk_id_sequence': str(session.spk_id_sequence),
    'distance_prime_spk': str(session.distance_prime_spk),
    'noise_path': str(session.noise_path),
    'absorption_applied': str(session.absorption_applied),
    'absorption_range': str(session.absorption_range), 
    'noise_gain_dB_applied': str(session.noise_gain_dB_applied),
    'noise_gain_dB_range': str(session.noise_gain_dB_range) }

    Pn = session.num_of_prime_spkrs
    Bn = session.num_of_bkg_spkrs
    
    for k in range(Bn):
        for l in range(Pn):
            index_str = 'distance_spk'+str(k+session.num_of_prime_spkrs+1)+'_ch'+str(l+1)
            info_dict[index_str] = str(session.distance_bgr_ch[k][l])

    ch_json.insert(0, info_dict)
    return ch_json


def json_modifier(current_time_sec, json_utt_list):
    new_start_sec_offset = current_time_sec - json_utt_list[0]['start']
    for ct, js in enumerate(json_utt_list):
        if json_utt_list[ct]['case'] == "success": 
            json_utt_list[ct]['start'] = json_utt_list[ct]['start'] + new_start_sec_offset
            json_utt_list[ct]['end'] = json_utt_list[ct]['end'] + new_start_sec_offset
            del json_utt_list[ct]['startOffset']
            del json_utt_list[ct]['endOffset']
    return json_utt_list


def read_LibriSpeech_data(verbose, LibriSpeech_directory, sess_dir_list, random_seed, number_of_spk_load):
    SR = 16000
    audio_folder_name = 'train-clean-100'
    json_folder_name = 'train-clean-100-json'

    data_path = LibriSpeech_directory + '/' + audio_folder_name + '/'
    json_path = json_folder_name + '/'
    data_list_mat = []  # list that contains the whole data set
    json_list_mat = []
    number_index_mat = []
    sprint(verbose, 'Loading Libri speech corpora... ', '')

    for count, cont in enumerate(sess_dir_list):  # This is a loop for each speaker
        sub_dir = data_path + cont
        sub_dir_list = os.listdir(sub_dir)
        utts_per_spk_count = 0
        audio_data_list = []
        json_list = []
        
        for ct, elem in enumerate(sub_dir_list):
            corp_dir = sub_dir + '/' + elem
            flac_filelist = glob.glob(corp_dir + '/*.flac')
            sprint(verbose, 'Loading Directory:', corp_dir)
            
            for flac_ct, flac_file_address in enumerate(flac_filelist):
                flac_only_file_name = flac_file_address.split('/')[-1]
                split_flac_fn = flac_only_file_name.split('-')
                json_file_address = json_path + split_flac_fn[0]  + '/'+ split_flac_fn[1] + '/' + flac_only_file_name[:-5] + '.json'
                if number_of_spk_load == -1:
                    pass
                if number_of_spk_load != -1 and utts_per_spk_count > number_of_spk_load:
                    break
                data = return_json_dict(json_file_address) 
                word_lens_mat, json_div_out, segment_durations = create_rand_utt(verbose, data, 0)
                audio_data_list.extend(read_and_divide_file(flac_file_address, segment_durations, SR)) 
                json_list.extend(json_div_out)
                utts_per_spk_count += 1
        
        # The list that contains the actual data.
        data_list_mat.append(audio_data_list)
        json_list_mat.append(json_list)
        
        # How many utterances each speaker has.
        number_index_mat.append(len(audio_data_list))

    return data_list_mat, json_list_mat, number_index_mat


def d2s(distance, SR):
    vc = 343
    return int(SR * (0.3 * distance/float(vc)))


def read_RIR(session):
    sprint(session.verbose, 'Creating RIR for the current session... ', '')
    np.random.seed(session.ses_idx)
    rir_bgr_ch = [ [None]*int(session.num_of_prime_spkrs) for x in range(session.num_of_bkg_spkrs) ]
    
    # width should be: width < 4 * distance
    width_room = 6
    rir_spk    , _ = get_sim_RIR(session.distance_prime_spk, width=width_room, absp_coeff=session.absorption_applied)
    session.rir_spk = IR_silence_remover(rir_spk, sample_margin=session.IR_sample_margin, IR_cut_thrs=session.IR_cut_thrs)
    # Add delay according to IR
    session.rir_spk = np.hstack((np.zeros(d2s(session.distance_prime_spk, session.SR)), session.rir_spk)) 
    
    Pn = session.num_of_prime_spkrs
    Bn = session.num_of_bkg_spkrs
    
    for k in range(Bn):
        for l in range(Pn):
            rir_bgr_ch[k][l], _ = get_sim_RIR(session.distance_bgr_ch[k][l], width=width_room, absp_coeff=session.absorption_applied)
            # print('rir_bgr_ch[',str(l),'][',str(k),']:', type(rir_bgr_ch[k][l]), np.shape(rir_bgr_ch[k][l]) )
            session.rir_bgr_ch[k][l] = IR_silence_remover(rir_bgr_ch[k][l], sample_margin=session.IR_sample_margin, IR_cut_thrs=session.IR_cut_thrs)
            session.rir_bgr_ch[k][l] = np.hstack((np.zeros(d2s(session.distance_bgr_ch[k][l], session.SR)), session.rir_bgr_ch[k][l]))
    

def random_spk_id_generator(dialogue_prob, number_of_spk_turns, number_of_dialogue_types):
    spk_mat = []
    for k in range(-1, number_of_dialogue_types, 1):
        spk_mat = spk_mat + [k] * int(dialogue_prob[k + 1] * 1000)
    # Random number that calls a value out of spk_mat which contains spk_ids with probability.
    spk_id_sequence = [spk_mat[np.random.randint(len(spk_mat))] for k in range(number_of_spk_turns)]
    return spk_id_sequence


def silence_generator(min_max_silence_length_sec, SR=16000):
    # Randomize the length
    length_in_sample = int(SR * min_max_silence_length_sec)
    return [0] * length_in_sample


def IR_silence_remover(RIR, sample_margin, IR_cut_thrs):
    max_idx = int(np.argmax(np.abs(RIR)))
    removed_RIR = RIR[np.max(max_idx - sample_margin, 0):]
    end_idx = len(removed_RIR)
    for k in range(len(removed_RIR))[::-1]:
        if np.abs(removed_RIR[k]) > IR_cut_thrs * np.max(np.abs(removed_RIR)):
            end_idx = k
            break
    removed_RIR = removed_RIR[:end_idx]
    return removed_RIR


def calculate_number_index(number_index_mat):
    length_index = np.argsort(number_index_mat)
    return length_index


def calculate_number_of_sess(number_index_mat, num_of_all_spkrs):
    number_of_sess = int(len(number_index_mat) / num_of_all_spkrs)
    return number_of_sess


def render_2ch_speech(section_wave_ts, RIR, distance_spk):
    near_field = 0.98 * normalize_wave(section_wave_ts)
    length_org = np.shape(section_wave_ts)[0]
    start_time = time.time()
    far_field_org = np.convolve(section_wave_ts, RIR)
    far_field = (1/float(distance_spk)) * normalize_wave(far_field_org[:length_org])
    return near_field, far_field


def render_bgr_speech(section_wave_ts, RIR1, RIR2, distance_bgr_ch1, distance_bgr_ch2):
    start_time = time.time()
    far_field_ch1 = np.convolve(section_wave_ts, RIR1)
    far_field_ch2 = np.convolve(section_wave_ts, RIR2)
    far_field_ch1 = far_field_ch1[:len(section_wave_ts)]
    far_field_ch2 = far_field_ch2[:len(section_wave_ts)]
    far_field_ch1 = (1/float(distance_bgr_ch1)) * normalize_wave(far_field_ch1)
    far_field_ch2 = (1/float(distance_bgr_ch2)) * normalize_wave(far_field_ch2)
    return far_field_ch1, far_field_ch2 


def render_ovl_speech(longer_near, longer_far, shorter_near, shorter_far):
    max_len = max(len(longer_near), len(shorter_near))
    min_len = min(len(longer_near), len(shorter_near))
    
    if max_len != min_len:
        rnd_idx = np.random.randint(0, min_len)
    elif max_len == min_len:
        rnd_idx = 0
    ch1_out, ch2_out = [np.zeros((len(longer_near) + rnd_idx)) for _ in range(2)]

    ch1_out[:len(shorter_near)] = shorter_near
    ch2_out[:len(shorter_near)] = shorter_far
    ch1_out[rnd_idx:(rnd_idx + len(longer_near))] += longer_far
    ch2_out[rnd_idx:(rnd_idx + len(longer_near))] += longer_near
    
    return ch1_out, ch2_out, rnd_idx


def read_noise_ts(session, dialogue_length_in_sample):
    SR = 16000
    dialogue_length = dialogue_length_in_sample * (1/float(SR))
    noise_directory = session.Noise_Data_directory + '/'

    session.noise_path = noise_directory + load_random_filename(session.verbose, 'QUT_noise_list', seed=session.ses_idx)
    sprint(session.verbose, 'Noise File:', session.noise_path)

    original_noise_SR = 48000
    y, _ = librosa.load(session.noise_path, offset=0.0, duration=dialogue_length, sr=original_noise_SR)
    session.noise_ts = librosa.resample(y, original_noise_SR, 16000)


def sig_power(sig):
    return np.sum(sig**2)/(np.shape(sig)[0])


def equalizer_gain(ch_list, noise_ts):
    ch_list = np.array(ch_list)
    eq_gain = np.sqrt(sig_power(ch_list)/sig_power(noise_ts))
    return eq_gain


def SNR_to_noise_gain(SNR):
    return 1/np.sqrt(10**(SNR/10))


def add_noise(ch_list, noise_ts, noise_gain_SNR):
    noise_gain = SNR_to_noise_gain(noise_gain_SNR) * equalizer_gain(ch_list, noise_ts)
    noise_ts = normalize_wave(noise_ts)
    len_ch = len(ch_list)
    len_noise = len(noise_ts)
    repeat_times = int(len_ch / len_noise) + 1
    noisy_out = []
    if len_ch > len_noise:
        for k in range(repeat_times):
            if k < repeat_times - 1:
                noisy_out.extend(
                    ch_list[(k * len_noise):((k + 1) * len_noise)] + noise_gain * noise_ts)
            elif k == (repeat_times - 1):
                rem_len = len_ch - (repeat_times - 1) * len_noise
                noisy_out.extend(ch_list[(
                    k * len_noise):(k * len_noise + rem_len)] + noise_gain * noise_ts[:rem_len])
    
    elif len_ch <= len_noise:
        noisy_out = ch_list + noise_gain * noise_ts[:len_ch]
    return noisy_out


def rttm_file_rec(dec_vec_ch, rttm_handle, file_id, spk_str, SR=16000):
    start_sec_idx = 0
    run_length = len(dec_vec_ch) - 1
    for vector_index in range(run_length):
        if dec_vec_ch[vector_index] == dec_vec_ch[vector_index + 1]:
            pass
        if vector_index > 0 and dec_vec_ch[vector_index] != dec_vec_ch[vector_index - 1]:
            start_sec_idx = vector_index * 1/float(SR)
            # print('start_sec_idx:', start_sec_idx)
        if dec_vec_ch[vector_index] != dec_vec_ch[vector_index + 1] or vector_index == run_length - 1:
            end_sec_idx = vector_index * 1/float(SR)
            spk_id_str = str(int(dec_vec_ch[vector_index]))
            duration = end_sec_idx - start_sec_idx
            if spk_id_str == '1':
                str_write = 'SPEAKER %s 1 %f %f <NA> <NA> %s <NA>\n' %(file_id, start_sec_idx, duration, spk_str)
                rttm_handle.write(str_write)


def write_rttm_file(spkr_decision_vec, json_output_directory, file_id, num_of_all_spkrs, spkr_list=True, SR=16000):
    dec_vec_ch1, dec_vec_ch2 = spkr_decision_vec
    dict_spk = {'-2':'SIL', '-1': 'SIL', '0':'SIL', '1': 'SP1', '2': 'SP2'}
    primary_spk = ['SP1', 'SP2']
    directory_for_time_stamp = json_output_directory + '/'
    if spkr_list == False:
        rttm_handle = open(directory_for_time_stamp + file_id + '.mdm.nnl' + '.rttm', 'w')
    
    if spkr_list == True:
        rttm_handle = open(directory_for_time_stamp + file_id + '_GT.mdm.nnl' + '.rttm', 'w')
        for ct, val in enumerate(primary_spk):
            if val != 'SIL':
                str_write = 'SPKR-INFO %s 1 <NA> <NA> <NA> unknown %s <NA>\n' %(file_id, val)
                rttm_handle.write(str_write)

    rttm_file_rec(dec_vec_ch1, rttm_handle, file_id, 'SP1', SR)
    rttm_file_rec(dec_vec_ch2, rttm_handle, file_id, 'SP2', SR)
    rttm_handle.close()
   
def json_list2dic(json_list):
    jlen = len(json_list)
    out_dict = {}
    for k, val in enumerate(json_list):
        if k == 0:
            out_dict['session_info'] = json_list[0]
        elif k != 0:
            out_dict[str(k)] = val
    return out_dict


def write_json_file(json_list, directory, file_id):
    json_ch1_dict = json_list2dic(json_list[0])
    json_ch2_dict = json_list2dic(json_list[1])
    with open(directory + '/' + file_id + '_ch1.json', 'w') as f:
        json.dump(json_ch1_dict, f, indent=4, sort_keys=True)
    with open(directory + '/' + file_id + '_ch2.json', 'w') as f:
        json.dump(json_ch2_dict, f, indent=4, sort_keys=True)


def make_visible(gt_data_out):
    gt_data_out[0] = [(0.25) * float(integral) for integral in gt_data_out[0]]
    gt_data_out[1] = [(0.25) * float(integral) for integral in gt_data_out[1]]
    return gt_data_out


def save_wavfile(wav_output_directory, file_string, ts_data, SR):
    if not os.path.exists(wav_output_directory):
        os.makedirs(wav_output_directory)
    sf.write(wav_output_directory + '/' + 'sample_session_' + file_string + '_ch1' +
             '.wav', np.array(ts_data[0]), SR)
    sf.write(wav_output_directory + '/' + 'sample_session_' + file_string + '_ch2' +  
             '.wav', np.array(ts_data[1]), SR)


def silence_fixer(gt_vec, cts, json_list, SR):
    th_s = 0.3
    gt_vec = np.array(gt_vec)
    for ct in range(len(json_list)):
        if ct > 0: 
            gap = json_list[ct]['start']-json_list[ct-1]['end']
            if gap > th_s:
                start = int((json_list[ct-1]['end'] - cts)*float(SR))
                end = int((json_list[ct]['start'] - cts)*float(SR))
                gt_vec[start:end] = 0
    return list(gt_vec)
    

def background_diag_gen(session, k_idx, spk_idx, current_time_sec):
    stt=time.time()
    spk_num_in = session.spk_id_sequence[k_idx]
    bsn = spk_num_in - 1 - session.num_of_prime_spkrs
    # print('spk', str(spk_num_in), 'spk_idx[2]:', spk_idx[spk_num_in-1])
    
    mic1_sig, mic2_sig = render_bgr_speech(session.data_list_mat[spk_idx[spk_num_in-1]][k_idx],
                                 session.rir_bgr_ch[bsn][0],
                                 session.rir_bgr_ch[bsn][1],
                                 session.distance_bgr_ch[bsn][0],
                                 session.distance_bgr_ch[bsn][1])

    ch1_ground_truth_buffer = [-1] * len(mic1_sig)
    ch2_ground_truth_buffer = [-1] * len(mic2_sig)
    current_time_sec += len(mic1_sig) * (1/float(session.SR)) 
    # print('Elapsed Time: ', (time.time() -stt))
    return mic1_sig, mic2_sig, ch1_ground_truth_buffer, ch2_ground_truth_buffer, current_time_sec 


def create_virtual_dialogue(session, json_list_mat, spk_idx):
    
    np.random.seed(session.ses_idx)

    ch1_list = []
    ch2_list = []
    ch1_ground_truth = []
    ch2_ground_truth = []
    ch1_ground_truth_buffer = []
    ch2_ground_truth_buffer = []
    ch1_json = []
    ch2_json = []
    
    prime_spk_list = list(range(0, session.num_of_prime_spkrs + 1))
    background_spk_list = list(range(session.num_of_prime_spkrs + 1, session.num_of_all_spkrs + 1))
   
    sprint(session.verbose, 'Primary Speaker:', str(prime_spk_list) + ' Interfering Speaker: ' + str(background_spk_list))
    
    current_time_sec = 0
    for k in range(session.number_of_spk_turns):
        # Take turns and speak according to "session.spk_id_sequence"

        ch1_json_buffer = []
        ch2_json_buffer = []
        
        sprint(session.verbose, 'Creating :', str(k+1)+'-th turn, dialogue type:'+str(session.spk_id_sequence[k]))

        if session.spk_id_sequence[k] == -1:  # silence
            silence = silence_generator(uni_rand_noseed(session.min_max_silence_length_sec),
                                        session.SR)
            mic1_sig = mic2_sig = silence

            ch1_ground_truth_buffer = [0] * len(silence)
            ch2_ground_truth_buffer = [0] * len(silence)
            current_time_sec += len(silence)*(1/float(session.SR))

        # spk1, spk2 both are speaking, overlapping section
        elif session.spk_id_sequence[k] == 0:
            stt = time.time()
            spk1_near, spk1_far = render_2ch_speech(session.data_list_mat[spk_idx[0]][k],
                                                    session.rir_spk,
                                                    session.distance_prime_spk)

            spk2_near, spk2_far = render_2ch_speech(session.data_list_mat[spk_idx[1]][k],
                                                    session.rir_spk,
                                                    session.distance_prime_spk)
            
            spk1_first = int(np.round(np.random.rand(1)[0]))  # Determine which one should be the first utterance.
            
            if len(spk1_near) >= len(spk2_near):
                mic2_sig, mic1_sig, rnd_idx = render_ovl_speech(
                    spk1_near, spk1_far, spk2_near, spk2_far)
                ch1_ground_truth_buffer = [-1] * rnd_idx + [1] * len(spk1_near)
                ch2_ground_truth_buffer = [1] * len(spk2_near) + [-1] * (len(spk1_near) - len(spk2_near) + rnd_idx)
                mod_json_ch1 = json_modifier(current_time_sec + rnd_idx * (1/float(session.SR)), json_list_mat[spk_idx[0]][k])
                mod_json_ch2 = json_modifier(current_time_sec, json_list_mat[spk_idx[1]][k])
                ch1_ground_truth_buffer = silence_fixer(ch1_ground_truth_buffer, current_time_sec, mod_json_ch1, session.SR)
                ch2_ground_truth_buffer = silence_fixer(ch2_ground_truth_buffer, current_time_sec, mod_json_ch2, session.SR)
                ch1_json_buffer = mod_json_ch1
                ch2_json_buffer = mod_json_ch2
                current_time_sec += len(spk1_near) * (1/float(session.SR)) + rnd_idx * (1/float(session.SR))

            elif len(spk1_near) < len(spk2_near):
                mic1_sig, mic2_sig, rnd_idx = render_ovl_speech(
                    spk2_near, spk2_far, spk1_near, spk1_far)
                ch1_ground_truth_buffer = [1] * len(spk1_near) + [-1] * (len(spk2_near) - len(spk1_near)+ rnd_idx)
                ch2_ground_truth_buffer = [-1] * rnd_idx + [1] * len(spk2_near)
                mod_json_ch1 = json_modifier(current_time_sec, json_list_mat[spk_idx[0]][k])
                mod_json_ch2 = json_modifier(current_time_sec + rnd_idx * (1/float(session.SR)), json_list_mat[spk_idx[1]][k])
                ch2_ground_truth_buffer = silence_fixer(ch2_ground_truth_buffer, current_time_sec, mod_json_ch2, session.SR)
                ch2_ground_truth_buffer = silence_fixer(ch2_ground_truth_buffer, current_time_sec, mod_json_ch2, session.SR)
                ch1_json_buffer = mod_json_ch1
                ch2_json_buffer = mod_json_ch2
                current_time_sec += len(spk2_near) * (1/float(session.SR))  + rnd_idx * (1/float(session.SR)) 
            # sprint(session.verbose, 'Elapsed Time: ', (time.time() -stt))

        elif session.spk_id_sequence[k] == 1:  # spk1 is speaking
            stt = time.time()
            # sprint(session.verbose, '1 spk1', 'spk_idx[1]:', spk_idx[0])
            mic1_sig, mic2_sig = render_2ch_speech(session.data_list_mat[spk_idx[0]][k],
                                                   session.rir_spk,
                                                   session.distance_prime_spk)

            ch1_ground_truth_buffer = [1] * len(mic1_sig)
            ch2_ground_truth_buffer = [-1] * len(mic1_sig)
            mod_json = json_modifier(current_time_sec, json_list_mat[spk_idx[0]][k])
            ch1_ground_truth_buffer = silence_fixer(ch1_ground_truth_buffer, current_time_sec, mod_json, session.SR)
            ch2_ground_truth_buffer = silence_fixer(ch2_ground_truth_buffer, current_time_sec, mod_json, session.SR)
            ch1_json_buffer = mod_json
            current_time_sec += len(mic1_sig) * (1/float(session.SR)) 
            # sprint(session.verbose, 'Elapsed Time: ', (time.time() -stt))

        elif session.spk_id_sequence[k] == 2:  # spk2 is speaking
            stt=time.time()
            # sprint(session.verbose, '2 spk2', 'spk_idx[2]:', spk_idx[1])
            mic2_sig, mic1_sig = render_2ch_speech(session.data_list_mat[spk_idx[1]][k],
                                                   session.rir_spk,
                                                   session.distance_prime_spk)
            
            ch1_ground_truth_buffer = [-1] * len(mic2_sig)
            ch2_ground_truth_buffer = [1] * len(mic2_sig)
            mod_json = json_modifier(current_time_sec, json_list_mat[spk_idx[1]][k])
            ch1_ground_truth_buffer = silence_fixer(ch1_ground_truth_buffer, current_time_sec, mod_json, session.SR)
            ch2_ground_truth_buffer = silence_fixer(ch2_ground_truth_buffer, current_time_sec, mod_json, session.SR)
            ch2_json_buffer = mod_json
            current_time_sec += len(mic2_sig) * (1/float(session.SR)) 
            # sprint(session.verbose, 'Elapsed Time: ', (time.time() -stt))

        elif session.spk_id_sequence[k] in background_spk_list:
            mic1_sig, mic2_sig, ch1_ground_truth_buffer, ch2_ground_truth_buffer, current_time_sec = background_diag_gen(session, k, spk_idx, current_time_sec)

        ch1_list.extend(mic1_sig)
        ch2_list.extend(mic2_sig)

        ch1_ground_truth.extend(ch1_ground_truth_buffer)
        ch2_ground_truth.extend(ch2_ground_truth_buffer)
        
        if session.spk_id_sequence[k] in prime_spk_list:
            ch1_json.extend(ch1_json_buffer)
            ch2_json.extend(ch2_json_buffer)
        # sprint(session.verbose, 'uttidx:', k + 1, '/', session.number_of_spk_turns, 'size sig1', np.shape(mic1_sig))
    
    return [ch1_list, ch2_list], [ch1_ground_truth, ch2_ground_truth], [ch1_json, ch2_json]


def dist_capper(dist_val, spk_dist, dist_prob_range):
    dist_val_out = dist_val
    if abs(dist_val - spk_dist) < dist_prob_range[0]:
        if (dist_val - spk_dist) >= 0:
            dist_val_out = spk_dist + dist_prob_range[0]
        else:
            dist_val_out = spk_dist - dist_prob_range[0]
    else:
        dist_val_out = dist_val
    return dist_val_out


def create_session(session):
    np.random.seed(session.random_seed)  # Fix the random seed.
    
    sprint(session.verbose, '\n--- Creating Session',  str(session.ses_idx + 1) + ' ---')
    
    number_of_spk_load = session.number_of_spk_turns  # Number of speakers 
    session.data_list_mat, json_list_mat, number_index_mat = read_LibriSpeech_data(session.verbose,
                                                            session.LibriSpeech_directory,
                                                            session.dir_list[session.ses_idx],
                                                            session.random_seed,
                                                            session.number_of_spk_load)
    
    session.distance_bgr_ch = [[None]*int(session.num_of_prime_spkrs) for x in range(session.num_of_bkg_spkrs)]
    session.rir_bgr_ch = [[None]*int(session.num_of_prime_spkrs) for x in range(session.num_of_bkg_spkrs)]
    session.distance_prime_spk = uni_rand_noseed(session.dist_prob_range_prime_spk)
    for k in range(session.num_of_bkg_spkrs):
        session.distance_bgr_ch[k][0] = dist_capper(uni_rand_noseed(session.dist_prob_range_bgr_spk), session.distance_prime_spk, session.dist_prob_range_bgr_spk)
        range_bgr_ch = [abs(session.distance_bgr_ch[k][0] - session.distance_prime_spk), min(session.distance_bgr_ch[k][0] + session.distance_prime_spk, session.dist_prob_range_bgr_spk[1])]
        session.distance_bgr_ch[k][1] = uni_rand_noseed(range_bgr_ch)
   
    sprint(session.verbose, 'Randomized Parameters', '')
    sprint(session.verbose, 'distance_prime_spk', str(round(session.distance_prime_spk,2)))
    
    for k in range(session.num_of_bkg_spkrs):
        for l in range(session.num_of_prime_spkrs):
            index_str = 'distance_spk'+str(k+session.num_of_prime_spkrs+1)+'_ch'+str(l+1)
            sprint(session.verbose, index_str, str(round(session.distance_bgr_ch[k][l],2)))
    
    # This "lengths_mat" contains the number of u tterance in a session per a speaker.
    # Since we want to balance the number of utterance of both speakers, we take minimum of these values.
    session.absorption_applied = round(uni_rand_noseed(session.absorption_range),2) 
                                        

    # Randomly load RIRs and noise from dataset.
    read_RIR(session)

    # How many utterances are in a session
    if session.number_of_spk_turns == -1:
        session.number_of_spk_turns = np.min(number_index_mat)

    # Set a randomized dialogue sequence
    session.number_of_dialogue_types = session.num_of_all_spkrs + 1
    spk_idx = [x for x in range(session.num_of_all_spkrs)]
    session.spk_id_sequence = random_spk_id_generator(session.dialogue_prob,
                                              session.number_of_spk_turns, 
                                              session.number_of_dialogue_types)  # Generate a random spk id sequence

    # Create a dialogue: outputs time series data and ground truth data
    ch_list, gt_list, json_list = create_virtual_dialogue(session, json_list_mat, spk_idx)

    # Add noise to time series data
    if session.noise:
        read_noise_ts(session, len(ch_list[0]))
        session.noise_gain_dB_applied = [round(uni_rand_noseed(session.noise_gain_dB_range),2),
                                         round(uni_rand_noseed(session.noise_gain_dB_range),2)]
        sprint(session.verbose, 'Noise_gain_dB_applied: ', str(session.noise_gain_dB_applied[0])+'dB for ch1'
                                       +str(session.noise_gain_dB_applied[1])+ 'dB for ch2')
        ch_list[0] = add_noise(ch_list[0], session.noise_ts, session.noise_gain_dB_applied[0])
        ch_list[1] = add_noise(ch_list[1], session.noise_ts, session.noise_gain_dB_applied[1])

    json_list[0] = add_seesion_info2json(json_list[0], session)
    json_list[1] = add_seesion_info2json(json_list[1], session)

    return ch_list, gt_list, json_list, session


def session_creator_wrapper(**kwargs):
    
    kwargs['num_of_bkg_spkrs'] = kwargs['num_of_all_spkrs'] - kwargs['num_of_prime_spkrs']
    kwargs['sr'] = 16000
    session_ET = []
   
    dir_list, number_of_sess = read_LibriSpeech_index(kwargs['verbose'], 
                                                      kwargs['librispeech_directory'], 
                                                      kwargs['num_of_all_spkrs'],
                                                      kwargs['num_of_prime_spkrs'])

    
    if kwargs['number_of_sess'] == -1:
        for_start = 0
        for_end = number_of_sess
    
    elif kwargs['number_of_sess'] == -2:
        for_start = kwargs['start']-1
        for_end = kwargs['end']
    
    elif kwargs['number_of_sess'] not in [-1, -2]:
        for_start = 0
        for_end = kwargs['number_of_sess']
        
    sprint(kwargs['verbose'], '\nStart Generating ',  str(for_end - for_start) + ' sessions.')

    for ses_idx in range(for_start, for_end):

        np.random.seed(ses_idx)
        stt_session = time.time()
        test_session = session(LibriSpeech_directory=kwargs['librispeech_directory'],
                               Noise_Data_directory=kwargs['noise_data_directory'],  # dir_list for the current session.
                               dir_list=dir_list,  # Directory list of speech files.
                               verbose=kwargs['verbose'],
                               ses_idx=ses_idx,  # session index of the loop.
                               SR=kwargs['sr'],  # SR of the original speech data set
                               random_seed=ses_idx,  # random seed for randomness
                               num_of_prime_spkrs=kwargs['num_of_prime_spkrs'],  # N of all the prime spkrs in a session.
                               num_of_all_spkrs=kwargs['num_of_all_spkrs'],  # N of all the speakers in a session.
                               num_of_bkg_spkrs=kwargs['num_of_bkg_spkrs'],  # N of all the background speakers in a session.
                               dialogue_prob=kwargs['dialogue_prob'],  # sil, ovl, spk1, spk2, spk3(bkg), spk4(bkg) 
                               dist_prob_range_bgr_spk=kwargs['dist_prob_range_bgr_spk'],
                               dist_prob_range_prime_spk=kwargs['dist_prob_range_prime_spk'],
                               number_of_spk_turns = kwargs['number_of_spk_turns'],  # Number of speaker turns
                               min_max_silence_length_sec=[0, 5],  # The minimum/maximum length of silence in second.
                               absorption_range=kwargs['absorption_range'],
                               noise=kwargs['noise'],  # Background noise
                               noise_gain_dB_range=kwargs['noise_gain_dB_range'],
                               IR_sample_margin=15,
                               IR_cut_thrs=0.00001)  # Background noise level

        # Create a session with given parameters.
        ts_data_out, gt_data_out, json_list, test_session = create_session(test_session)
        
        save_wavfile(kwargs['wav_output_directory'], str(ses_idx + 1), ts_data_out, kwargs['sr'])  # Save to wavefiles.
        file_id = kwargs['file_id'] + str(ses_idx + 1) 
        write_rttm_file(gt_data_out, kwargs['wav_output_directory'], file_id, kwargs['num_of_all_spkrs'], spkr_list=True, SR=test_session.SR)
        write_json_file(json_list, kwargs['wav_output_directory'], file_id) 
        ET = time.time() - stt_session
        session_ET.append(ET)
        sprint(kwargs['verbose'], 'Deleting Memory...', '')

        del ts_data_out, gt_data_out, json_list
        del test_session
