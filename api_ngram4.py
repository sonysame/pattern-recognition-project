import os
from tqdm import tqdm
from collections import defaultdict
import csv
import pandas as pd
import numpy as np
import pickle
import random
import time
def main():
	_2gram_feature_api = []
	with open("_2gram_api_feature_selected", 'r') as reader:
		data=reader.read()
		lines=data.split(",\n")[:-1]
	for i in tqdm(range(len(lines)), mininterval=1):
		tmp = lines[i].replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
		if lines[i] not in _2gram_feature_api:
			_2gram_feature_api.append(tmp)

	print(len(_2gram_feature_api))

	_3gram_feature_api = []
	with open("_3gram_api_feature_selected", 'r') as reader:
		data = reader.read()
		lines = data.split(",\n")[:-1]
	for i in tqdm(range(len(lines)),mininterval=1):
		tmp = lines[i].replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
		if lines[i] not in _3gram_feature_api:
			_3gram_feature_api.append(tmp)

	print(len(_3gram_feature_api))

	_4gram_feature_api = []
	with open("_4gram_api_feature_selected", 'r') as reader:
		data = reader.read()
		lines = data.split(",\n")[:-1]
	for i in tqdm(range(len(lines)),mininterval=1):
		tmp = lines[i].replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
		if lines[i] not in _4gram_feature_api:
			_4gram_feature_api.append(tmp)

	print(len(_4gram_feature_api))

	ori_path = "/Users/heeyeon/Desktop/pattern_recognition/extract_API/"

	path_file= [ori_path+("/benign"),ori_path+("/malware")]
	total_file=[os.listdir(path_file[0])[:len(os.listdir(path_file[0]))//10*8],os.listdir(path_file[1])[:len(os.listdir(path_file[1]))//10*8]]
	total_file_test = [os.listdir(path_file[0])[len(os.listdir(path_file[0])) // 10 * 8:],
					   os.listdir(path_file[1])[len(os.listdir(path_file[1])) // 10 * 8:]]

	start_time=time.time()
	_2gram_dataset=np.zeros((len(total_file[0])+len(total_file[1]),len(_2gram_feature_api)+1))
	for category_idx in range(len(total_file)):
		for i in tqdm(range(len(total_file[category_idx])), mininterval=1):
			target_file = total_file[category_idx][i]
			w=0.
			with open(path_file[category_idx] + "/" + target_file, 'r') as reader:
				api = []
				data = reader.read()
				lines = data.split("\n")[1:-1]
				for line in lines:
					api.append(line.split(',')[1])
			_2gram = {}

			for j in _2gram_feature_api:
				_2gram[j] = 0

			for j in range(len(api) - 1):
				key="'" + api[j] + "','" + api[j + 1] + "'"
				if key in _2gram:
					_2gram[key]+=1
					w+=1
			tmp=np.round(np.array(list(_2gram.values()))/(w+0.1),4).tolist()
			tmp.append(category_idx)
			if category_idx==0:
				_2gram_dataset[i]=np.array(tmp)
			elif category_idx==1:
				_2gram_dataset[(len(os.listdir(path_file[0]))//10*8)+i]=np.array(tmp)
	column_data=_2gram_feature_api+["category"]
	_2gram_selected_df=pd.DataFrame(_2gram_dataset, columns=column_data)
	_2gram_selected_df.to_pickle('2gram_selected.pkl')

	print(time.time()-start_time)

	start_time=time.time()
	_2gram_dataset_test = np.zeros((len(total_file_test[0]) + len(total_file_test[1]), len(_2gram_feature_api) + 1))
	for category_idx in range(len(total_file_test)):
		for i in tqdm(range(len(total_file_test[category_idx])), mininterval=1):
			target_file = total_file_test[category_idx][i]
			w = 0.
			with open(path_file[category_idx] + "/" + target_file, 'r') as reader:
				api = []
				data = reader.read()
				lines = data.split("\n")[1:-1]
				for line in lines:
					api.append(line.split(',')[1])
			_2gram = {}
			for j in _2gram_feature_api:
				_2gram[j] = 0

			for j in range(len(api) - 1):
				key = "'" + api[j] + "','" + api[j + 1] + "'"
				if key in _2gram:
					_2gram[key] += 1
					w += 1
			tmp = np.round(np.array(list(_2gram.values())) / (w+0.1), 4).tolist()
			tmp.append(category_idx)

			if category_idx==0:
				_2gram_dataset_test[i]=np.array(tmp)
			elif category_idx==1:
				_2gram_dataset_test[len(os.listdir(path_file[0]))-(len(os.listdir(path_file[0]))//10*8)+i]=np.array(tmp)
	column_data = _2gram_feature_api+["category"]
	_2gram_selected_test_df = pd.DataFrame(_2gram_dataset_test, columns=column_data)
	_2gram_selected_test_df.to_pickle('2gram_selected_test.pkl')
	print(time.time() - start_time)

	start_time = time.time()
	_3gram_dataset=np.zeros((len(total_file[0])+len(total_file[1]),len(_3gram_feature_api)+1))
	for category_idx in range(len(total_file)):
		for i in tqdm(range(len(total_file[category_idx])), mininterval=1):
			target_file = total_file[category_idx][i]
			w=0.
			with open(path_file[category_idx] + "/" + target_file, 'r') as reader:
				api = []
				data = reader.read()
				lines = data.split("\n")[1:-1]
				for line in lines:
					api.append(line.split(',')[1])
			_3gram = {}

			for j in _3gram_feature_api:
				_3gram[j] = 0

			for j in range(len(api) - 2):
				key = "'" + api[j] + "','" + api[j + 1] + "','" + api[j + 2] + "'"
				if key in _3gram:
					_3gram[key]+=1
					w+=1
			tmp=np.round(np.array(list(_3gram.values()))/(w+0.1),4).tolist()
			tmp.append(category_idx)
			if category_idx==0:
				_3gram_dataset[i]=np.array(tmp)
			elif category_idx==1:
				_3gram_dataset[(len(os.listdir(path_file[0]))//10*8)+i]=np.array(tmp)
	column_data=_3gram_feature_api+["category"]
	_3gram_selected_df=pd.DataFrame(_3gram_dataset, columns=column_data)
	_3gram_selected_df.to_pickle('3gram_selected.pkl')

	print(time.time() - start_time)

	start_time = time.time()
	_3gram_dataset_test = np.zeros((len(total_file_test[0]) + len(total_file_test[1]), len(_3gram_feature_api) + 1))
	for category_idx in range(len(total_file_test)):
		for i in tqdm(range(len(total_file_test[category_idx])), mininterval=1):
			target_file = total_file_test[category_idx][i]
			w = 0.
			with open(path_file[category_idx] + "/" + target_file, 'r') as reader:
				api = []
				data = reader.read()
				lines = data.split("\n")[1:-1]
				for line in lines:
					api.append(line.split(',')[1])
			_3gram = {}
			for j in _3gram_feature_api:
				_3gram[j] = 0

			for j in range(len(api) - 2):
				key = "'" + api[j] + "','" + api[j + 1] + "','" + api[j + 2] + "'"
				if key in _3gram:
					_3gram[key] += 1
					w += 1
			tmp = np.round(np.array(list(_3gram.values())) / (w+0.1), 4).tolist()
			tmp.append(category_idx)

			if category_idx==0:
				_3gram_dataset_test[i]=np.array(tmp)
			elif category_idx==1:
				_3gram_dataset_test[len(os.listdir(path_file[0]))-(len(os.listdir(path_file[0]))//10*8)+i]=np.array(tmp)
	column_data = _3gram_feature_api+["category"]
	_3gram_selected_test_df = pd.DataFrame(_3gram_dataset_test, columns=column_data)
	_3gram_selected_test_df.to_pickle('3gram_selected_test.pkl')
	print(time.time() - start_time)

	start_time = time.time()
	_4gram_dataset=np.zeros((len(total_file[0])+len(total_file[1]),len(_4gram_feature_api)+1))
	for category_idx in range(len(total_file)):
		for i in tqdm(range(len(total_file[category_idx])), mininterval=1):
			target_file = total_file[category_idx][i]
			w=0.
			with open(path_file[category_idx] + "/" + target_file, 'r') as reader:
				api = []
				data = reader.read()
				lines = data.split("\n")[1:-1]
				for line in lines:
					api.append(line.split(',')[1])
			_4gram = {}

			for j in _4gram_feature_api:
				_4gram[j] = 0

			for j in range(len(api) - 3):
				key = "'" + api[j] + "','" + api[j + 1] + "','" + api[j + 2] + "','" + api[j + 3] + "'"

				if key in _4gram:
					_4gram[key]+=1
					w+=1
			tmp=np.round(np.array(list(_4gram.values()))/(w+0.1),4).tolist()
			tmp.append(category_idx)
			if category_idx==0:
				_4gram_dataset[i]=np.array(tmp)
			elif category_idx==1:
				_4gram_dataset[(len(os.listdir(path_file[0]))//10*8)+i]=np.array(tmp)
	column_data=_4gram_feature_api+["category"]
	_4gram_selected_df=pd.DataFrame(_4gram_dataset, columns=column_data)
	_4gram_selected_df.to_pickle('4gram_selected.pkl')

	print(time.time() - start_time)

	start_time = time.time()
	_4gram_dataset_test = np.zeros((len(total_file_test[0]) + len(total_file_test[1]), len(_4gram_feature_api) + 1))
	for category_idx in range(len(total_file_test)):
		for i in tqdm(range(len(total_file_test[category_idx])), mininterval=1):
			target_file = total_file_test[category_idx][i]
			w = 0.
			with open(path_file[category_idx] + "/" + target_file, 'r') as reader:
				api = []
				data = reader.read()
				lines = data.split("\n")[1:-1]
				for line in lines:
					api.append(line.split(',')[1])
			_4gram = {}
			for j in _4gram_feature_api:
				_4gram[j] = 0

			for j in range(len(api) - 3):
				key = "'" + api[j] + "','" + api[j + 1] + "','" + api[j + 2] + "','" + api[j + 3] + "'"

				if key in _4gram:
					_4gram[key] += 1
					w += 1
			tmp = np.round(np.array(list(_4gram.values())) / (w+0.1), 4).tolist()
			tmp.append(category_idx)

			if category_idx==0:
				_4gram_dataset_test[i]=np.array(tmp)
			elif category_idx==1:
				_4gram_dataset_test[len(os.listdir(path_file[0]))-(len(os.listdir(path_file[0]))//10*8)+i]=np.array(tmp)
	column_data = _4gram_feature_api+["category"]
	_4gram_selected_test_df = pd.DataFrame(_4gram_dataset_test, columns=column_data)
	_4gram_selected_test_df.to_pickle('4gram_selected_test.pkl')

	print(time.time() - start_time)

if __name__=="__main__":
	main()