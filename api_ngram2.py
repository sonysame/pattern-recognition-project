import os
from tqdm import tqdm
import csv
import pandas as pd
import numpy as np
import time


def main():
    total_2gram_api = []
    with open("2gram_api.csv", 'r') as reader:
        data = reader.read()
        lines = data.split("\n")[1:-1]
    for i in lines:
        tmp = i.split(')')[0].replace('(', '').replace('"', '').replace(' ', '')
        total_2gram_api.append(tmp)
    total_2gram = {}
    for i in total_2gram_api:
        total_2gram[i] = [0, 0, 0, 0]

    total_3gram_api = []
    with open("3gram_api.csv", 'r') as reader:
        data = reader.read()
        lines = data.split("\n")[1:-1]
    for i in lines:
        tmp = i.split(')')[0].replace('(', '').replace('"', '').replace(' ', '')
        total_3gram_api.append(tmp)
    total_3gram = {}
    for i in total_3gram_api:
        total_3gram[i] = [0, 0, 0, 0]

    total_4gram_api = []
    with open("4gram_api.csv", 'r') as reader:
        data = reader.read()
        lines = data.split("\n")[1:-1]
    for i in lines:
        tmp = i.split(')')[0].replace('(', '').replace('"', '').replace(' ', '')
        total_4gram_api.append(tmp)
    total_4gram = {}
    for i in total_4gram_api:
        total_4gram[i] = [0, 0, 0, 0]

    ori_path = "/Users/heeyeon/Desktop/pattern_recognition/extract_API/"

    mode = ['benign', 'malware']
    total_file=[os.listdir(ori_path+mode[0]), os.listdir(ori_path+mode[1])]
    for mode_index in range(len(mode)):
        path_file = ori_path
        path_file += mode[mode_index]
        print(path_file)
        start_time = time.time()
        for i in tqdm(range(len(total_file[mode_index])//10*8), mininterval=1):
            target_file = total_file[mode_index][i]
            with open(path_file + "/" + target_file, 'r') as reader:
                api = []
                data = reader.read()
                lines = data.split("\n")[1:-1]
                for line in lines:
                    api.append(line.split(',')[1])
            _2gram = {}
            _3gram = {}
            _4gram = {}
            for j in total_2gram_api:
                _2gram[j] = False
            for j in total_3gram_api:
                _3gram[j] = False
            for j in total_4gram_api:
                _4gram[j] = False

            for j in range(len(api) - 3):
                total_2gram["'" + api[j] + "','" + api[j + 1] + "'"][2 * mode_index] += 1
                if _2gram["'" + api[j] + "','" + api[j + 1] + "'"] == False:
                    _2gram["'" + api[j] + "','" + api[j + 1] + "'"] = True

                total_3gram["'" + api[j] + "','" + api[j + 1] + "','" + api[j + 2] + "'"][2 * mode_index] += 1
                if _3gram["'" + api[j] + "','" + api[j + 1] + "','" + api[j + 2] + "'"] == False:
                    _3gram["'" + api[j] + "','" + api[j + 1] + "','" + api[j + 2] + "'"] = True

                total_4gram["'" + api[j] + "','" + api[j + 1] + "','" + api[j + 2] + "','" + api[j + 3] + "'"][
                    2 * mode_index] += 1
                if _4gram["'" + api[j] + "','" + api[j + 1] + "','" + api[j + 2] + "','" + api[j + 3] + "'"] == False:
                    _4gram["'" + api[j] + "','" + api[j + 1] + "','" + api[j + 2] + "','" + api[j + 3] + "'"] = True

            if (len(api) >= 2):
                total_2gram["'" + api[len(api) - 2] + "','" + api[len(api) - 1] + "'"][2 * mode_index] += 1
                if _2gram["'" + api[len(api) - 2] + "','" + api[len(api) - 1] + "'"] == False:
                    _2gram["'" + api[len(api) - 2] + "','" + api[len(api) - 1] + "'"] = True

                if (len(api) >= 3):
                    total_2gram["'" + api[len(api) - 3] + "','" + api[len(api) - 2] + "'"][2 * mode_index] += 1
                    if _2gram["'" + api[len(api) - 3] + "','" + api[len(api) - 2] + "'"] == False:
                        _2gram["'" + api[len(api) - 3] + "','" + api[len(api) - 2] + "'"] = True

                    total_3gram["'" + api[len(api) - 3] + "','" + api[len(api) - 2] + "','" + api[len(api) - 1] + "'"][
                        2 * mode_index] += 1
                    if _3gram[
                        "'" + api[len(api) - 3] + "','" + api[len(api) - 2] + "','" + api[len(api) - 1] + "'"] == False:
                        _3gram["'" + api[len(api) - 3] + "','" + api[len(api) - 2] + "','" + api[
                            len(api) - 1] + "'"] = True

            for j in _2gram:
                if _2gram[j] == True:
                    total_2gram[j][2 * mode_index + 1] += 1

            for j in _3gram:
                if _3gram[j] == True:
                    total_3gram[j][2 * mode_index + 1] += 1

            for j in _4gram:
                if _4gram[j] == True:
                    total_4gram[j][2 * mode_index + 1] += 1
        print(time.time() - start_time)
    key2List = total_2gram.keys()
    _2benign_tf = np.array(list(total_2gram.values()))[:, 0]
    _2benign_df = np.array(list(total_2gram.values()))[:, 1]
    _2malware_tf = np.array(list(total_2gram.values()))[:, 2]
    _2malware_df = np.array(list(total_2gram.values()))[:, 3]

    key3List = total_3gram.keys()
    _3benign_tf = np.array(list(total_3gram.values()))[:, 0]
    _3benign_df = np.array(list(total_3gram.values()))[:, 1]
    _3malware_tf = np.array(list(total_3gram.values()))[:, 2]
    _3malware_df = np.array(list(total_3gram.values()))[:, 3]

    key4List = total_4gram.keys()
    _4benign_tf = np.array(list(total_4gram.values()))[:, 0]
    _4benign_df = np.array(list(total_4gram.values()))[:, 1]
    _4malware_tf = np.array(list(total_4gram.values()))[:, 2]
    _4malware_df = np.array(list(total_4gram.values()))[:, 3]

    _2benign_calc_tf = np.round(_2benign_tf / np.sum(_2benign_tf), 4)
    _2_binary_benign_calc_idf = np.round(((len(total_file[0])//10*8)+(len(total_file[1])//10*8)) / (_2benign_df + _2malware_df + 0.1), 4)
    _2_binary_benign_tf_idf = np.round(np.multiply(_2benign_tf, _2_binary_benign_calc_idf), 4)

    _2malware_calc_tf = np.round(_2malware_tf / np.sum(_2malware_tf), 4)
    _2_binary_malware_calc_idf = np.round(((len(total_file[0])//10*8)+(len(total_file[1])//10*8)) / (_2benign_df + _2malware_df + 0.1), 4)
    _2_binary_malware_tf_idf = np.round(np.multiply(_2malware_tf, _2_binary_malware_calc_idf), 4)
    #_2_binary_malware_tf_idf = np.round(np.multiply(np.multiply(_2malware_tf, _2malware_df),_2_binary_malware_calc_idf), 4)
    #_2_binary_malware_tf_idf = np.round(np.multiply(np.multiply(_2malware_tf, np.log2(_2malware_df+0.1)),_2_binary_malware_calc_idf), 4)
    #_2_binary_malware_tf_idf = np.round(np.multiply(np.multiply(_2malware_tf, np.log10(_2malware_df+0.1)),_2_binary_malware_calc_idf), 4)

    _3benign_calc_tf = np.round(_3benign_tf / np.sum(_3benign_tf), 4)
    _3_binary_benign_calc_idf = np.round(((len(total_file[0])//10*8)+(len(total_file[1])//10*8)) / (_3benign_df + _3malware_df + 0.1), 4)
    _3_binary_benign_tf_idf = np.round(np.multiply(_3benign_tf, _3_binary_benign_calc_idf), 4)

    _3malware_calc_tf = np.round(_3malware_tf / np.sum(_3malware_tf), 4)
    _3_binary_malware_calc_idf = np.round(((len(total_file[0])//10*8)+(len(total_file[1])//10*8)) / (_3benign_df + _3malware_df + 0.1), 4)
    _3_binary_malware_tf_idf = np.round(np.multiply(_3malware_tf, _3_binary_malware_calc_idf), 4)
    #_3_binary_malware_tf_idf = np.round(np.multiply(np.multiply(_3malware_tf, _3malware_df),_3_binary_malware_calc_idf), 4)
    #_3_binary_malware_tf_idf = np.round(np.multiply(np.multiply(_3malware_tf, np.log2(_3malware_df+0.1)),_3_binary_malware_calc_idf), 4)
    #_3_binary_malware_tf_idf = np.round(np.multiply(np.multiply(_3malware_tf, np.log10(_3malware_df+0.1)),_3_binary_malware_calc_idf), 4)

    _4benign_calc_tf = np.round(_4benign_tf / np.sum(_4benign_tf), 4)
    _4_binary_benign_calc_idf = np.round(((len(total_file[0])//10*8)+(len(total_file[1])//10*8)) / (_4benign_df + _4malware_df + 0.1), 4)
    _4_binary_benign_tf_idf = np.round(np.multiply(_4benign_tf, _4_binary_benign_calc_idf), 4)

    _4malware_calc_tf = np.round(_4malware_tf / np.sum(_4malware_tf), 4)
    _4_binary_malware_calc_idf = np.round(((len(total_file[0])//10*8)+(len(total_file[1])//10*8)) / (_4benign_df + _4malware_df + 0.1), 4)
    _4_binary_malware_tf_idf = np.round(np.multiply(_4malware_tf, _4_binary_malware_calc_idf), 4)
    #_4_binary_malware_tf_idf = np.round(np.multiply(np.multiply(_4malware_tf, _4malware_df),_4_binary_malware_calc_idf), 4)
    #_4_binary_malware_tf_idf = np.round(np.multiply(np.multiply(_4malware_tf, np.log2(_4malware_df+0.1)),_4_binary_malware_calc_idf), 4)
    #_4_binary_malware_tf_idf = np.round(np.multiply(np.multiply(_4malware_tf, np.log10(_4malware_df+0.1)),_4_binary_malware_calc_idf), 4)

    _2malware_calc_tf = np.round(_2malware_tf / np.sum(_2malware_tf), 4)
    _3malware_calc_tf = np.round(_3malware_tf / np.sum(_3malware_tf), 4)
    _4malware_calc_tf = np.round(_4malware_tf / np.sum(_4malware_tf), 4)

    # 2gram-benign
    data = np.stack((_2benign_tf, _2malware_tf,_2benign_calc_tf, _2malware_calc_tf, _2benign_df, _2malware_df, _2_binary_benign_tf_idf, _2_binary_malware_tf_idf), axis=1)
    data_df = pd.DataFrame(data, columns=["2gram_benign_TF", "2gram_malware_TF","2gram_benign_calc_TF", "2gram_malware_calc_TF", "2gram_benign_DF",
                                          "2gram_malware_DF", "2gram_binary_benign_TF-IDF","2gram_binary_malware_TF-IDF"],
                           index=total_2gram_api)
    data_df.to_pickle('2gram_api_result.pkl')
    rows1 = zip(key2List, _2benign_tf.tolist(), _2malware_tf.tolist(),_2benign_calc_tf.tolist(), _2malware_calc_tf.tolist(), _2benign_df.tolist(),
                _2malware_df.tolist(), _2_binary_benign_tf_idf.tolist(), _2_binary_malware_tf_idf.tolist())
    with open('2gram_api_result.csv', 'w', encoding='utf-8') as f:
        w1 = csv.writer(f)
        w1.writerow(["2gram_api", "2gram_benign_TF", "2gram_malware_TF", "2gram_benign_calc_TF", "2gram_malware_calc_TF", "2gram_benign_DF", "2gram_malware_DF",
                     "2gram_binary_benign_TF-IDF", "2gram_binary_malware_TF-IDF"])
        for row in rows1:
            w1.writerow(row)

    # 3gram-benign
    data = np.stack((_3benign_tf, _3malware_tf,_3benign_calc_tf, _3malware_calc_tf, _3benign_df, _3malware_df, _3_binary_benign_tf_idf, _3_binary_malware_tf_idf), axis=1)
    data_df = pd.DataFrame(data, columns=["3gram_benign_TF", "3gram_malware_TF","3gram_benign_calc_TF", "3gram_malware_calc_TF", "3gram_benign_DF",
                                          "3gram_malware_DF", "3gram_binary_benign_TF-IDF","3gram_binary_malware_TF-IDF"],
                           index=total_3gram_api)
    data_df.to_pickle('3gram_api_result.pkl')
    rows1 = zip(key3List, _3benign_tf.tolist(), _3malware_tf.tolist(),_3benign_calc_tf.tolist(), _3malware_calc_tf.tolist(), _3benign_df.tolist(),
                _3malware_df.tolist(), _3_binary_benign_tf_idf.tolist(), _3_binary_malware_tf_idf.tolist())
    with open('3gram_api_result.csv', 'w', encoding='utf-8') as f:
        w1 = csv.writer(f)
        w1.writerow(["3gram_api", "3gram_benign_TF", "3gram_malware_TF","3gram_benign_calc_TF", "3gram_malware_calc_TF", "3gram_benign_DF", "3gram_malware_DF",
                     "3gram_binary_benign_TF-IDF", "3gram_binary_malware_TF-IDF"])
        for row in rows1:
            w1.writerow(row)
    # 4gram-benign
    data = np.stack((_4benign_tf, _4malware_tf,_4benign_calc_tf, _4malware_calc_tf, _4benign_df, _4malware_df, _4_binary_benign_tf_idf, _4_binary_malware_tf_idf), axis=1)
    data_df = pd.DataFrame(data, columns=["4gram_benign_TF", "4gram_malware_TF", "4gram_benign_calc_TF", "4gram_malware_calc_TF","4gram_benign_DF",
                                          "4gram_malware_DF", "4gram_binary_benign_TF-IDF","4gram_binary_malware_TF-IDF"],
                           index=total_4gram_api)
    data_df.to_pickle('4gram_api_result.pkl')
    rows1 = zip(key4List, _4benign_tf.tolist(), _4malware_tf.tolist(),_4benign_calc_tf.tolist(), _4malware_calc_tf.tolist(), _4benign_df.tolist(),
                _4malware_df.tolist(), _4_binary_benign_tf_idf.tolist(), _4_binary_malware_tf_idf.tolist())
    with open('4gram_api_result.csv', 'w', encoding='utf-8') as f:
        w1 = csv.writer(f)
        w1.writerow(["4gram_api", "4gram_benign_TF", "4gram_malware_TF", "4gram_benign_calc_TF", "4gram_malware_calc_TF","4gram_benign_DF", "4gram_malware_DF",
                     "4gram_binary_benign_TF-IDF", "4gram_binary_malware_TF-IDF"])
        for row in rows1:
            w1.writerow(row)


if __name__ == "__main__":
    main()