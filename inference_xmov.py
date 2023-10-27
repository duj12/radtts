import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import argparse
import os
import json
import numpy as np
import torch
from torch.cuda import amp
from scipy.io.wavfile import write

from radtts_xmov import RADTTS
from data_xmov import Data, DataCollate, load_wav_to_torch
from torch.utils.data import DataLoader
from common import update_params

from hifigan_xmov import Generator
from hifigan_env import AttrDict
from hifigan_denoiser import Denoiser

TextGridFilterSymbol={
    "",
}
LAUGH_PHONEMES_SINGLE=[
    "HHH15", "AAA15", "EEE15", "III15", "OOO15", "NNN15", "YYY15"
]
SPECIAL_PHONEMES_SINGLE=[
    "AA5", "AH5", "EN5", "EY5", "AE5",
    "AA7", "AH7", "EN7", "EY7", "AE7",
    "AA8", "AH8", "EN8", "EY8", "AE8",
    "AA9", "AH9", "EN9", "EY9", "AE9",
    "AA10", "AH10", "EN10", "EY10", "AE10",
    "AA11", "AH11", "EN11", "EY11", "AE11",
    "AA12", "AH12", "EN12", "EY12", "AE12",
    "AA13", "AH13", "EN13", "EY13", "AE13",
]
SPECIAL_PHONEMES_DOUBLE=[
    ["L", "AA5"],["N", "AA5"],["N", "AH5"],["Y", "AO5"],["Y", "OW5"],["W", "AA5"],["L", "EY5"],
    ["L", "AA7"],["N", "AA7"],["N", "AH7"],["Y", "AO7"],["Y", "OW7"],["W", "AA7"],["L", "EY7"],
    ["L", "AA8"],["N", "AA8"],["N", "AH8"],["Y", "AO8"],["Y", "OW8"],["W", "AA8"],["L", "EY8"],
    ["L", "AA9"],["N", "AA9"],["N", "AH9"],["Y", "AO9"],["Y", "OW9"],["W", "AA9"],["L", "EY9"],
    ["L", "AA10"],["N", "AA10"],["N", "AH10"],["Y", "AO10"],["Y", "OW10"],["W", "AA10"],["L", "EY10"],
    ["L", "AA11"],["N", "AA11"],["N", "AH11"],["Y", "AO11"],["Y", "OW11"],["W", "AA11"],["L", "EY11"],
    ["L", "AA12"],["N", "AA12"],["N", "AH12"],["Y", "AO12"],["Y", "OW12"],["W", "AA12"],["L", "EY12"],
    ["L", "AA13"],["N", "AA13"],["N", "AH13"],["Y", "AO13"],["Y", "OW13"],["W", "AA13"],["L", "EY13"],
]


def lines_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def load_vocoder(vocoder_path, config_path, to_cuda=True):
    with open(config_path) as f:
        data_vocoder = f.read()
    config_vocoder = json.loads(data_vocoder)
    h = AttrDict(config_vocoder)
    if 'blur' in vocoder_path:
        config_vocoder['gaussian_blur']['p_blurring'] = 0.5
    else:
        if 'gaussian_blur' in config_vocoder:
            config_vocoder['gaussian_blur']['p_blurring'] = 0.0
        else:
            config_vocoder['gaussian_blur'] = {'p_blurring': 0.0}
            h['gaussian_blur'] = {'p_blurring': 0.0}

    state_dict_g = torch.load(vocoder_path, map_location='cpu')['generator']

    # load hifigan
    vocoder = Generator(h)
    vocoder.load_state_dict(state_dict_g)
    denoiser = Denoiser(vocoder)
    if to_cuda:
        vocoder.cuda()
        denoiser.cuda()
    vocoder.eval()
    denoiser.eval()

    return vocoder, denoiser


def parse_data_from_batch(batch):
    mel = batch['mel']
    speaker_ids = batch['speaker_ids']
    text = batch['text']
    tone = batch['tone']
    lang = batch['lang']
    in_lens = batch['input_lengths']
    out_lens = batch['output_lengths']
    attn_prior = batch['attn_prior']
    f0 = batch['f0']
    voiced_mask = batch['voiced_mask']
    p_voiced = batch['p_voiced']
    energy_avg = batch['energy_avg']
    audiopaths = batch['audiopaths']
    if attn_prior is not None:
        attn_prior = attn_prior.cuda()
    if f0 is not None:
        f0 = f0.cuda()
    if voiced_mask is not None:
        voiced_mask = voiced_mask.cuda()
    if p_voiced is not None:
        p_voiced = p_voiced.cuda()
    if energy_avg is not None:
        energy_avg = energy_avg.cuda()

    mel, speaker_ids = mel.cuda(), speaker_ids.cuda()
    text, tone, lang = text.cuda(), tone.cuda(), lang.cuda()
    in_lens, out_lens = in_lens.cuda(), out_lens.cuda()

    return (mel, speaker_ids, text, tone, lang,
            in_lens, out_lens, attn_prior, f0,
            voiced_mask, p_voiced, energy_avg, audiopaths)


def infer(radtts_path, vocoder_path, vocoder_config_path, text_path, speaker,
          speaker_text, speaker_attributes, sigma, sigma_tkndur, sigma_f0,
          sigma_energy, f0_mean, f0_std, energy_mean, energy_std,
          token_dur_scaling, denoising_strength, n_takes, output_dir, use_amp,
          plot, seed, use_dp=False, do_skip=False, add_p_before_haha=False):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    vocoder, denoiser = load_vocoder(vocoder_path, vocoder_config_path)
    radtts = RADTTS(**model_config).cuda()
    radtts.enable_inverse_cache() # cache inverse matrix for 1x1 invertible convs

    checkpoint_dict = torch.load(radtts_path, map_location='cpu')
    state_dict = checkpoint_dict['state_dict']
    radtts.load_state_dict(state_dict, strict=False)
    radtts.eval()
    print("Loaded checkpoint '{}')" .format(radtts_path))
    if output_dir is None:
        output_dir = os.path.dirname(radtts_path)
        output_dir = os.path.join(output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    ignore_keys = ['training_files', 'validation_files', 'test_files']
    if 'test_files' in data_config:
        test_files = 'test_files'
    else:
        test_files = 'validation_files'
    testset = Data(
        data_config[test_files],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))

    # Option 1: give test_files in configs
    if 'test_files' in data_config:
        testdataloader = DataLoader(testset, num_workers=1, shuffle=False,
                              sampler=None, batch_size=16,
                              pin_memory=False, drop_last=False,
                              collate_fn=DataCollate())
        n_batches = len(testdataloader)
        testdata = testset.load_data(data_config[test_files])

        for i, batch in enumerate(testdataloader):
            phoslists = batch['phonelists']
            (mel_gt, speaker_ids, text, tone, lang,
                in_lens, out_lens, attn_prior,
                f0, voiced_mask, p_voiced, energy_avg,
                audiopaths) = parse_data_from_batch(batch)

            with amp.autocast(use_amp):
                with torch.no_grad():
                    outputs = radtts.infer(
                        speaker_ids, text, tone, lang,
                        sigma, sigma_tkndur, sigma_f0,
                        sigma_energy, token_dur_scaling,
                        token_duration_max=100,
                        speaker_id_text=speaker_ids,
                        speaker_id_attributes=speaker_ids,
                        f0_mean=f0_mean, f0_std=f0_std,
                        energy_mean=energy_mean,
                        energy_std=energy_std,
                        mel_gt=mel_gt, attn_prior=attn_prior,
                        in_lens=in_lens, mel_lens=out_lens,
                    )

                    mel = outputs['mel']
                    lens = outputs['lens']
                    audio = vocoder(mel).float()
                    for j in range(len(phoslists)):
                        phos_list = phoslists[j]
                        audio_len = lens[j] * data_config['hop_length']
                        audio_denoised = denoiser(
                            audio[j,:,:audio_len], strength=denoising_strength).float()
                        audio_denoised = audio_denoised[0][0].cpu().numpy()
                        audio_denoised = audio_denoised / np.max(
                            np.abs(audio_denoised))

                        audiopath = audiopaths[j]
                        if audiopath is not None:
                            suffix_path = audiopath.split('/')
                            path_len = min(len(suffix_path), 4)
                            suffix_path = suffix_path[-path_len:]
                            suffix_path = '_'.join(suffix_path).replace('.wav','')
                        else:
                            suffix_path = "{}_{}_{}_durscaling{}_sigma{}_sigmatext{}_sigmaf0{}_sigmaenergy{}".format(
                                i, j, speaker, token_dur_scaling, sigma,
                                sigma_tkndur, sigma_f0,
                                sigma_energy)

                        write("{}/{}_denoised_{}.wav".format(
                            output_dir, suffix_path, denoising_strength),
                            data_config['sampling_rate'], audio_denoised)

                        if mel_gt is not None and outputs['attn'] is not None:

                            if use_dp:  # do not use the attn from mel and text
                                # use the duration predictor result
                                durations = outputs['dur'][j]
                            else:
                                attn = outputs['attn']
                                durations = attn[j, 0].sum(0, keepdim=True)[0]
                                durations = (durations + 0.5).floor().int()
                            dur_second = durations * data_config[
                                'hop_length'] / data_config['sampling_rate']
                            dur_second = list(dur_second.cpu().numpy())
                            # phos_clear_list = clear_pho(phos_list)
                            get_and_save_TextGrid(dur_second, phos_list,
                                                  filename=f"{output_dir}/"
                                                           f"{suffix_path}.TextGrid")

            if plot:
                fig, axes = plt.subplots(2, 1, figsize=(10, 6))
                axes[0].plot(outputs['f0'].cpu().numpy()[0], label='f0')
                axes[1].plot(outputs['energy_avg'].cpu().numpy()[0],
                             label='energy_avg')
                for ax in axes:
                    ax.legend(loc='best')
                plt.tight_layout()
                fig.savefig(
                    "{}/{}_features.png".format(output_dir, suffix_path))
                plt.close('all')

    # Option 2: give a textfile, include text|spk|lang|audio
    if text_path is not None:
        text_list = lines_to_list(text_path)
    else:
        text_list = []

    if speaker:
        speaker_id = testset.get_speaker_id(speaker).cuda()
        speaker_id_text, speaker_id_attributes = speaker_id, speaker_id
    if speaker_text is not None:
        speaker_id_text = testset.get_speaker_id(speaker_text).cuda()
    if speaker_attributes is not None:
        speaker_id_attributes = testset.get_speaker_id(
            speaker_attributes).cuda()

    # single batch iter on text_list
    all_num, all_shift, ave_shift = 0, 0, 0
    special_all_num, special_all_shift, special_ave_shift = 0, 0, 0
    laugh_all_num, laugh_all_shift, laugh_ave_shift = 0, 0, 0
    for i, text in enumerate(text_list):
        if text.startswith("#"):
            continue
        print("processing: {}/{}: {}".format(i, len(text_list), text))

        text = text.split('|')
        spk = None
        lang_id = 0
        audiopath = None
        mel, attn_prior = None, None
        in_lens, out_lens = None, None
        phos_list, phos_with_dur = None, None

        # process each line
        if len(text) == 2:
            text, spk = text[0], text[1]
        elif len(text) == 3:
            text, spk, lang_id = text[0], text[1], int(text[2])
        elif len(text) == 4:
            text, spk, lang_id, audiopath = text[0:4]
            audio, sampling_rate = load_wav_to_torch(audiopath)
            mel_gt = testset.get_mel(audio)
            lang_id = int(lang_id)

        # process text
        if text.startswith('["') and text.endswith('"]'):  # convert to list
            text = text[2:-2]
            text = text.split('", "')
            phos_list = text
        elif text.endswith('.TextGrid'):  # support textgrid input
            from textgrid import TextGrid
            # 创建TextGrid对象并加载文件
            tg = TextGrid()
            tg.read(text)

            # 获取第2个Tier（音素信息在第2个Tier中）
            tier = tg[1]
            # 读取音素序列和时间戳
            phos_list = []
            phos_with_dur = []
            for interval in tier:
                item = {}
                key = interval.mark
                if key in TextGridFilterSymbol:
                    continue
                val = float(interval.maxTime) - float(interval.minTime)
                item[key] = val
                phos_list.append(key)
                phos_with_dur.append(val)  # only store dur

        if spk is not None:
            speaker = spk
            speaker_id = testset.get_speaker_id(speaker).cuda()
            speaker_id_text, speaker_id_attributes = speaker_id, speaker_id
        if add_p_before_haha:
            phos_list_add_p, add_p_pos = [], []
            for p in phos_list:
                phos_list_add_p.append(p)
                add_p_pos.append(0)
                if p in LAUGH_PHONEMES_SINGLE:
                    phos_list_add_p.append("/")
                    add_p_pos.append(1)
            text, tone, lang = testset.get_text(phos_list_add_p, language_id=lang_id)
        else:
            text, tone, lang = testset.get_text(phos_list, language_id=lang_id)
        text = text.cuda()[None]
        tone = tone.cuda()[None]
        lang = lang.cuda()[None]

        if mel_gt is not None:
            mel_gt = mel_gt.cuda()[None]
            in_lens = torch.LongTensor(1).cuda()
            out_lens = torch.LongTensor(1).cuda()
            for i in range(1):
                in_lens[i] = text.size(1)
                out_lens[i] = mel_gt.size(2)
            attn_prior = testset.get_attention_prior(
                in_lens[0].item(), out_lens[0].item())
            attn_prior = attn_prior.cuda()[None]

        for take in range(n_takes):
            with amp.autocast(use_amp):
                with torch.no_grad():
                    outputs = radtts.infer(
                        speaker_id, text, tone, lang,
                        sigma, sigma_tkndur, sigma_f0,
                        sigma_energy, token_dur_scaling,
                        token_duration_max=100,
                        speaker_id_text=speaker_id_text,
                        speaker_id_attributes=speaker_id_attributes,
                        f0_mean=f0_mean, f0_std=f0_std,
                        energy_mean=energy_mean,
                        energy_std=energy_std,
                        mel_gt=mel_gt, attn_prior=attn_prior,
                        in_lens=in_lens, mel_lens=out_lens
                    )

                    mel = outputs['mel']
                    audio = vocoder(mel).float()[0]
                    audio_denoised = denoiser(
                        audio, strength=denoising_strength)[0].float()

                    audio = audio[0].cpu().numpy()
                    audio_denoised = audio_denoised[0].cpu().numpy()
                    audio_denoised = audio_denoised / np.max(np.abs(audio_denoised))

                    if audiopath is not None:
                        suffix_path = audiopath.split('/')
                        path_len = min(len(suffix_path), 1)
                        suffix_path = suffix_path[-path_len:]
                        suffix_path = '_'.join(suffix_path).replace('.wav','')
                    else:
                        suffix_path = "{}_{}_{}_durscaling{}_sigma{}_sigmatext{}_sigmaf0{}_sigmaenergy{}".format(
                            i, take, speaker, token_dur_scaling,
                            sigma, sigma_tkndur, sigma_f0,
                            sigma_energy)

                    write("{}/{}_denoised_{}.wav".format(
                        output_dir, suffix_path, denoising_strength),
                        data_config['sampling_rate'], audio_denoised)

                    if mel_gt is not None and outputs['attn'] is not None:
                        if use_dp:  # do not use the attn from mel and text
                            # use the duration predictor result
                            durations = outputs['dur'][0]
                        else:
                            attn = outputs['attn']
                            durations = attn[0, 0].sum(0, keepdim=True)[0]
                            durations = (durations + 0.5).floor().int()

                        dur_second = durations * data_config['hop_length'] / data_config['sampling_rate']
                        dur_second = list(dur_second.cpu().numpy())
                        # phos_clear_list = clear_pho(phos_list)
                        if add_p_before_haha:
                            dur_second_without_p = []
                            for i, d in enumerate(dur_second):
                                if add_p_pos[i] == 0:
                                    dur_second_without_p.append(d)
                                # elif add_p_pos[i] == 1:
                                #     dur_second_without_p[-1] += d
                            dur_second = dur_second_without_p

                        get_and_save_TextGrid(dur_second, phos_list,
                                              filename=f"{output_dir}/"
                                                       f"{suffix_path}.TextGrid")

                        if phos_with_dur is not None:  # measure the dur_pred
                            assert len(phos_with_dur) == len(dur_second), \
                                "duration of reference and prediction should be equal."
                            total_shift, total_len = 0, 0
                            s_shift, s_num = 0, 0
                            l_shift, l_num = 0, 0
                            for i, (pho, r, p) in enumerate(zip(phos_list, phos_with_dur, dur_second)):
                                if pho.startswith('#') and do_skip:
                                    continue
                                if i > 0 and [phos_list[i-1], phos_list[i]] in SPECIAL_PHONEMES_DOUBLE:
                                    print(phos_list[i - 1], phos_list[i])
                                    s_shift += abs(r-p)
                                    s_shift += abs(phos_with_dur[i-1] - dur_second[i-1])
                                    s_num += 2
                                elif phos_list[i] in SPECIAL_PHONEMES_SINGLE:
                                    print(phos_list[i])
                                    s_shift += abs(r - p)
                                    s_num += 1
                                elif phos_list[i] in LAUGH_PHONEMES_SINGLE:
                                    print(phos_list[i])
                                    l_shift += abs(r - p)
                                    l_num += 1
                                total_shift += abs(r-p)
                                total_len += 1

                            print(f"total {total_len} duration shift of {suffix_path} is {total_shift}.")
                            print(f"total {s_num} special phonemes duration shift is {s_shift}.")
                            print(f"total {l_num} laugh phonemes duration shift is {l_shift}.")
                            all_shift += total_shift
                            all_num += total_len
                            special_all_num += s_num
                            special_all_shift += s_shift
                            laugh_all_num += l_num
                            laugh_all_shift += l_shift

            if plot:
                fig, axes = plt.subplots(2, 1, figsize=(10, 6))
                axes[0].plot(outputs['f0'].cpu().numpy()[0], label='f0')
                axes[1].plot(outputs['energy_avg'].cpu().numpy()[0], label='energy_avg')
                for ax in axes:
                    ax.legend(loc='best')
                plt.tight_layout()
                fig.savefig("{}/{}_features.png".format(output_dir, suffix_path))
                plt.close('all')

    if all_num != 0:
        ave_shift = all_shift / all_num
        print(f"total {all_num} duration shift is {all_shift}. "
              f"average is {ave_shift}")
    if special_all_num != 0:
        special_ave_shift = special_all_shift / special_all_num
        print(
            f"total {special_all_num} special phonemes duration shift is {special_all_shift}. "
            f"average is {special_ave_shift}")
    if laugh_all_num != 0:
        laugh_ave_shift = laugh_all_shift / laugh_all_num
        print(
            f"total {laugh_all_num} laugh phonemes duration shift is {laugh_all_shift}. "
            f"average is {laugh_ave_shift}")


# 存储时间戳为TextGrid
def get_and_save_TextGrid(dur_ts, pho, filename, dump_json=False):
    from textgrid import TextGrid, IntervalTier
    time_range = {}
    beginning = 0
    # 创建 TextGrid 对象
    tg = TextGrid()
    # 创建 IntervalTier 对象
    interval_tier = IntervalTier(name='phones')
    for i in range(len(dur_ts)):
        pho_item = pho[i]

        ending = beginning + dur_ts[i]
        start_time = beginning
        end_time = ending
        label = pho_item
        interval_tier.add(start_time, end_time, label)
        time_range[i + 1] = {"pho": pho_item,
                             "xmin": beginning,
                             "xmax": ending,
                             "dur": dur_ts[i]}
        beginning = ending

    tg.append(interval_tier)
    tg.write(filename)
    if dump_json:
        json_name = filename.replace(".TextGrid", ".json")
        with open(json_name, "w") as write_file:
            json.dump(time_range, write_file, indent=4)


# 清理 Phonemes 中的分词和韵律信息
def clear_pho(pho_list):
    pho_clear_list = []
    for pho_item in pho_list:
        if pho_item not in ["#1", "#2", "#3", "#4", "/"]:
            pho_clear_list.append(pho_item)
    return pho_clear_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file config')
    parser.add_argument('-k', '--config_vocoder', type=str, help='vocoder JSON file config')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-r', '--radtts_path', type=str)
    parser.add_argument('-v', '--vocoder_path', type=str)
    parser.add_argument('-t', '--text_path', type=str)
    parser.add_argument('-s', '--speaker', type=str)
    parser.add_argument('--speaker_text', type=str, default=None)
    parser.add_argument('--speaker_attributes', type=str, default=None)
    parser.add_argument('-d', '--denoising_strength', type=float, default=0.0)
    parser.add_argument('-o', "--output_dir", type=str, default=None)
    parser.add_argument("--sigma", default=0.8, type=float, help="sampling sigma for decoder")
    parser.add_argument("--sigma_tkndur", default=0.666, type=float, help="sampling sigma for duration")
    parser.add_argument("--sigma_f0", default=1.0, type=float, help="sampling sigma for f0")
    parser.add_argument("--sigma_energy", default=1.0, type=float, help="sampling sigma for energy avg")
    parser.add_argument("--f0_mean", default=0.0, type=float)
    parser.add_argument("--f0_std", default=0.0, type=float)
    parser.add_argument("--energy_mean", default=0.0, type=float)
    parser.add_argument("--energy_std", default=0.0, type=float)
    parser.add_argument("--token_dur_scaling", default=1.00, type=float)
    parser.add_argument("--n_takes", default=1, type=int)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--use_dp", action="store_true")
    parser.add_argument("--do_skip", action="store_true")
    parser.add_argument("--add_p_before_haha", action="store_true")
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    update_params(config, args.params)

    data_config = config["data_config"]
    global model_config
    model_config = config["model_config"]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    infer(args.radtts_path, args.vocoder_path, args.config_vocoder,
          args.text_path, args.speaker, args.speaker_text,
          args.speaker_attributes, args.sigma, args.sigma_tkndur, args.sigma_f0,
          args.sigma_energy, args.f0_mean, args.f0_std, args.energy_mean,
          args.energy_std, args.token_dur_scaling, args.denoising_strength,
          args.n_takes, args.output_dir, args.use_amp, args.plot, args.seed,
          args.use_dp, args.do_skip, args.add_p_before_haha)
