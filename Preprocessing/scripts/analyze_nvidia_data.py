import json
import numpy as np
import os 
import glob
import seaborn as sns
import matplotlib.pyplot as plt

def find_files_by_pattern(path, pattern, recursive=True):
	"""
	PURPOSE: Imports ALL files from a chosen folder based on a given pattern
	INPUTS
	-------------------------------------------------------------
	pattern : list of strings with particular patterns, including filetype!
			ex: ["_patched",".csv"] will pull any csv files under filepath with the string "_patched" in its file name.

	filepath : string, path for where to search for files
			ex: "/users/<username>/folder"

	recursive : boolean, True if you wish for the search for files to be recursive under filepath.
	"""
	# generate pattern finding
	fpatterns = ["**{}".format(x) for i,x in enumerate(pattern)]
	if path[-1]!="/":
		path = path + "/"
	all_file_names = set()
	for fpattern in fpatterns:
		file_paths = glob.iglob(path + fpattern, recursive=recursive)
		for file_path in file_paths:
			# skip hidden files
			if file_path[0] == ".":
				continue
			file_name = os.path.basename(file_path)
			all_file_names.add(file_name)

	all_file_names = [file_name for file_name in all_file_names]
	all_file_names.sort()  # sort based on name

	return all_file_names

#####################################################################
##################### USER SPECIFIED PATH ###########################
#####################################################################

json_directory = "/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/HandE_annotations/annotations"

#####################################################################
#####################################################################

json_files = find_files_by_pattern(path=json_directory, pattern=['.json']) #grab the json files

all_classes = np.array([])
print("loading annotation data")

for jfile in json_files:
	json_file_path = os.path.join(json_directory, jfile)
	with open(json_file_path) as f:
		current_anno_data=json.load(f)
		
	all_classes = np.concatenate([all_classes, np.array(current_anno_data['features']['class'])], axis=0)

print('done, calculating unique numbers.')

classes, counts = np.unique(all_classes, return_counts = True)

print("done, generating plot.")
data = np.array([[x,counts[np.argwhere(classes==x).flatten()[0]]] if x in classes else [x,0] for x in range(20)])

fig, ax = plt.subplots(1)

sns.barplot(x = data[:,0], y = data[:,1], ax=ax)

xlabels = ['missed','G3','G4 cribriform','G4 poor-form', 'G4 fused', 'G4 glomeruloid', 
	   'G5 single cells', 'G5 sheets','G5 cords','G5 solid nests','G5 necrosis','neoplasia', 'benign',
	   'atrophy', 'nerve', 'vein', 'inflammation', 'stroma', 'bad seg', 'artifact'] #labels, according to word doc

ax.set_title("Class Count")
ax.set_ylabel("Counts")
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels, rotation=90)
plt.subplots_adjust(bottom=0.25)

plt.savefig("/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Pilot/scripts/class_plot.pdf",format='pdf')
plt.close()
print("Complete.")