import os
from tqdm import tqdm
from collections import defaultdict
import csv
import pandas as pd

_2gram_api_df = pd.read_pickle("./2gram_api_result.pkl")
#_2gram_api_df.info()
_2gram_api_sorted_df=_2gram_api_df.sort_values(by=['2gram_binary_malware_TF-IDF'], ascending=False)

f=open("./_2gram_api_feature_selected",'w')
f.write(str(_2gram_api_sorted_df.index.tolist()).replace(" ","\n"))
f.close


_3gram_api_df = pd.read_pickle("./3gram_api_result.pkl")
#_3gram_api_df.info()
_3gram_api_sorted_df=_3gram_api_df.sort_values(by=['3gram_binary_malware_TF-IDF'], ascending=False)

f=open("./_3gram_api_feature_selected",'w')
f.write(str(_3gram_api_sorted_df.index.tolist()).replace(" ","\n"))
f.close


_4gram_api_df = pd.read_pickle("./4gram_api_result.pkl")
#_4gram_api_df.info()
_4gram_api_sorted_df=_4gram_api_df.sort_values(by=['4gram_binary_malware_TF-IDF'], ascending=False)

f=open("./_4gram_api_feature_selected",'w')
f.write(str(_4gram_api_sorted_df.index.tolist()).replace(" ","\n"))
f.close



print(_2gram_api_df.shape, _3gram_api_df.shape, _4gram_api_df.shape)