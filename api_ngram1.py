import os
from tqdm import tqdm
from collections import defaultdict
import csv
import time
def main():
	ori_path=os.getcwd()
	#total_1gram=defaultdict(lambda: 0)
	total_2gram=defaultdict(lambda: 0)
	total_3gram=defaultdict(lambda: 0)
	total_4gram=defaultdict(lambda: 0)
	mode = ['benign', 'malware']

	ori_path = "/Users/heeyeon/Desktop/pattern_recognition/extract_API/"

	for mode_index in range(len(mode)):
		path_file = ori_path
		path_file+=(mode[mode_index])
		print(path_file)
		total_file=os.listdir(path_file)
		start_time=time.time()
		for i in tqdm(range(len(total_file)), mininterval=1):
			target_file=total_file[i]
			with open(path_file+"/"+target_file, 'r') as reader:
				api=[]
				data = reader.read()
				lines = data.split("\n")[1:-1]
				for line in lines:
					api.append(line.split(',')[1])
			for j in range(len(api)-3):
				#total_1gram[(api[j])] += 1
				total_2gram[(api[j], api[j + 1])] += 1
				total_3gram[(api[j], api[j + 1], api[j + 2])] += 1
				total_4gram[(api[j], api[j + 1], api[j + 2], api[j + 3])] += 1

			if(len(api)>=2):
				#total_1gram[]
				total_2gram[(api[len(api) - 2], api[len(api) - 1])] += 1
				if(len(api)>=3):
					total_2gram[(api[len(api) - 3], api[len(api) - 2])] += 1
					total_3gram[(api[len(api) - 3], api[len(api) - 2], api[len(api) - 1])] += 1

		print(time.time()-start_time)

	key2List=total_2gram.keys()
	value2List=total_2gram.values()
	rows2=zip(key2List, value2List)

	key3List = total_3gram.keys()
	value3List = total_3gram.values()
	rows3 = zip(key3List, value3List)

	key4List=total_4gram.keys()
	value4List=total_4gram.values()
	rows4=zip(key4List, value4List)

	with open('2gram_api.csv', 'w', encoding='utf-8') as f:
		w2=csv.writer(f)
		w2.writerow(["2-gram","frequency"])
		for row in rows2:
			w2.writerow(row)

	with open('3gram_api.csv', 'w', encoding='utf-8') as f:
		w3=csv.writer(f)
		w3.writerow(["3-gram","frequency"])
		for row in rows3:
			w3.writerow(row)

	with open('4gram_api.csv', 'w', encoding='utf-8') as f:
		w4=csv.writer(f)
		w4.writerow(["4-gram","frequency"])
		for row in rows4:
			w4.writerow(row)


if __name__=="__main__":
	main()