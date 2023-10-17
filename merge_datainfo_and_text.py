# !/usr/bin/env python3
# encoding: utf-8

import os
import json
import glob


def load_and_merge_info(data_info, textdict):
    new_datainfo = {}
    for speaker, info in data_info.items():
        spk = speaker.split('|')[0]
        sid = int(speaker.split('|')[1])
        total_dur, file_list = info
        new_filelist = []
        for file in file_list:
            filename, dur = file[0], file[1]
            newfile = file
            if spk in textdict and filename in textdict[spk]:
                newfile.append({"text": textdict[spk][filename]})

                new_filelist.append(newfile)
            else:
                print(f"{spk} not in the text sequence, OR, "
                      f"{filename} don't have a text sequence. "
                      f"audio of speaker {spk} will not be used!!")
        if len(new_filelist) > 0:
            new_datainfo[speaker] = [total_dur, new_filelist]
    return new_datainfo


def load_text(textdir, filter_key):
    glob_path = os.path.join(textdir + '/*_1011.json')
    file_paths = sorted(list(glob.glob(glob_path, recursive=True)))
    if len(file_paths) == 0:
        print(f"cannot find any files in {glob_path}")
    print("len(file_paths):", len(file_paths))
    phos_merged_dict = {}
    for file_path in file_paths:
        speaker_id = file_path.split("/")[-1].split("_FrontEnd")[0]
        with open(file_path, 'r', encoding='utf8') as fid:
            info_dict = json.load(fid)
        filter_info = {}
        for utt, textdict in info_dict.items():
            filter_info[utt] = textdict[filter_key]
        phos_merged_dict[speaker_id] = filter_info.copy()
    return phos_merged_dict


if __name__ == "__main__":
    used_seq_name = "format_all"   # 这个是训练时使用的序列
    base_dir = "/data/megastore/Datasets/55data/TTS"
    dirs = ['ENTTS', "LTTS", "MSTTS_CN", "MSTTS_EN", "NTTS"]
    subsets = ["valid", "train"]
    for dir in dirs:
        curdir = os.path.join(base_dir, dir)
        textdir = os.path.join(curdir, "ori_text_data")
        textdict = load_text(textdir, used_seq_name)
        for subset in subsets:
            infopath = os.path.join(curdir, f"{subset}set_info_{dir}.json")
            datainfo = json.load(open(infopath, 'r', encoding='utf-8'))
            merged_info = load_and_merge_info(datainfo, textdict)
            new_infopath = f"{curdir}/{subset}set_{dir}.json"
            json.dump(
                merged_info,
                open(new_infopath, 'w', encoding='utf-8'),
                indent=2,
                sort_keys=True,
                ensure_ascii=False
            )



