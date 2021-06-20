import os
from tqdm import tqdm
from collections import defaultdict
import csv
import time
import numpy as np

file_info=[[],[],[],[],[],[]]
api_info=[[],[],[],[],[],[]]
sample_info=[[],[],[],[],[],[]]

def main():
    ori_path="/Users/heeyeon/Desktop/pattern_recognition/ProcessId/"
    family=['ProcessIdCleanHippo', 'ProcessIdCleanZero', 'ProcessIdClean', 'ProcessIdCleanPippo', 'ProcessIdVirusShare500','ProcessIdVirusShare1000']

    for i in range(len(family)):
        total_file=os.listdir(ori_path+family[i])
        #print(total_file)
        if ".DS_Store" in total_file:
            total_file.remove(".DS_Store")
        for j in total_file:
            file_info[i].append(family[i]+"/"+j)

    for i in range(len(file_info)):
        for j in range(len(file_info[i])):
            sample_info[i].append([])
            api_info[i].append([])
            tmp=os.listdir(ori_path+file_info[i][j])
            if ori_path+file_info[i][j]+".DS_Store" in tmp:
                tmp.remove(".DS_Store")
            for k in tmp:
                sample_info[i][j].append(ori_path+file_info[i][j]+"/"+k)
            #print(sample_info[i])

    for i in range(len(sample_info)):
        start_time = time.time()
        for j in tqdm(range(len(sample_info[i])), mininterval=1):
            #print("GOOD", sample_info[i][j])
            for k in range(len(sample_info[i][j])):
                #print(k)
                f=open(sample_info[i][j][k],'r', encoding="windows-1252")
                tmp=[]
                while True:
                    try:
                        line = f.readline()
                        if "Time=" in line:
                            time_info=int(line.split("Time=")[1].split(',')[0])
                            tmp.append((time_info, sample_info[i][j][k].split("/")[-1].split(".")[0]))
                        #api_info[i][j].append(line)
                        if not line:
                            tmp=list(set(tmp))
                            api_info[i][j]+=tmp
                            break
                    except:
                        print(sample_info[i][j][k])
                        tmp=list(set(tmp))
                        api_info[i][j]+=tmp
                        break
                f.close()
            api_info[i][j].sort(key=lambda x:x[0])

        print(time.time()-start_time)

    os.system("rm -rf /Users/heeyeon/Desktop/pattern_recognition/extract_API")
    os.system("mkdir /Users/heeyeon/Desktop/pattern_recognition/extract_API")
    os.system("mkdir /Users/heeyeon/Desktop/pattern_recognition/extract_API/benign")
    os.system("mkdir /Users/heeyeon/Desktop/pattern_recognition/extract_API/malware")

    ori_path="/Users/heeyeon/Desktop/pattern_recognition/extract_API/"

    for i in range(len(api_info)):
        start_time=time.time()
        if(i<4):
            path_file=ori_path+"benign/"
        else:
            path_file=ori_path+"malware/"
        for j in tqdm(range(len(api_info[i])), mininterval=1):
            with open(path_file+family[i]+"_"+file_info[i][j].split("/")[1]+".csv",'w', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(["time", "API"])
                for apis in api_info[i][j]:
                    w.writerow(apis)
        print(time.time()-start_time)

if __name__=="__main__":
    main()