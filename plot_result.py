import _pickle as pickle 
import numpy as np 
import seaborn as sns
import matplotlib  
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help = "dir to the log files")
args = parser.parse_args()

log_files = [os.path.join(args.dir, p) for p in sorted(os.listdir(args.dir))]

all_result = []
all_steps = []
names = []
for log in log_files:
	if log.endswith(".pck"):
		names.append(log.split('.')[0].split('/')[2])
		file = open(log, 'rb')
		steps = []
		results = []
		while True:
			try:
				s, r = pickle.load(file)
				steps.append(s)
				results.append(r)
			except:
				file.close()
				break
		all_result.append(np.array(results))
		all_steps.append(np.array(steps))

for i in range(len(all_result)):
	steps = all_steps[i]
	results = all_result[i]
	df = pd.DataFrame(results.transpose())
	df.columns = steps
	sns.lineplot(x = 'variable', y = 'value', data = df.melt())
plt.legend(names)
plt.show()
