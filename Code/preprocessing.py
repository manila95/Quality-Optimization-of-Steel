# %matplotlib inline
import numpy as np
import scipy.io as io
import pandas as pd
import cPickle as cp
from sklearn.preprocessing import normalize
# import seaborn as sns


# Load data

data = cp.load(open("../Data/all_data_together.pkl"))

coils = data.keys()

columns = data["COIL_88540102"].keys()
new_df = pd.DataFrame(columns=columns)

for i in columns:
    temp = []
    out_temp = []
    for j in data.keys():
        temp.append(data[j][i])
    new_df[i] = temp

cooling_seq = np.array(new_df["COOLING_SEQ"])

for i, _ in enumerate(cooling_seq):
    cooling_seq[i] = cooling_seq[i][0].astype(int)

cooling_sequence = []
length_sequence = []
for cool in cooling_seq:
    new_cool = cool
    seq_length = cool.shape[0]
    length_sequence += [seq_length]
    cool_embed = []
    for i in range(160):
        temp_vector = np.zeros(3)
        if i < seq_length:
            temp_vector[new_cool[i]] = 1
        cool_embed.append(temp_vector)
    cooling_sequence.append(cool_embed)

# cooling_sequence = np.array(cooling_sequence)
    
cooling_sequence = np.array(cooling_sequence)

new_df = new_df.drop(['AIM_QLTY', 'COOLING_SEQ', "HR_ID"], axis = 1)

def normalization(list_parameter):
    list_parameter = np.array(list_parameter).astype(float)
    mean = np.mean(list_parameter)
    std = np.std(list_parameter)
    temp = []
    for parameter in list_parameter:
        temp.append((parameter - mean)/std)
    return temp

for col in new_df.columns:
    new_df[col] = normalization(new_df[col])

output_variables = new_df[['EL', 'UTS', 'LYS']]
new_df = new_df.drop(['EL', 'UTS', 'LYS'], axis = 1)

# output_variables = new_df[['EL', 'UTS', 'LYS']]
output_variables = np.array(output_variables)
output_variables = np.array(output_variables).astype(float)

list_chemical = ["NI", "NB", "SI", "TI", "C", "B", "AL", "P", "S", "V", "CR", "CU", "MN", "N"]
chemical_parameters = new_df[list_chemical]

process_parameters = new_df.drop(list_chemical, axis = 1)

chemical_parameters = np.array(chemical_parameters).astype(float)
process_parameters = np.array(process_parameters).astype(float)
# cooling_seq = cooling_seq.astype(int)

trainset = {}
trainset["process_parameters"] = process_parameters[:30000]
trainset["chemical_parameters"] = chemical_parameters[:30000]
trainset["cooling_sequence"] = cooling_sequence[:30000]
trainset["output_parameters"] = output_variables[:30000]
trainset["sequence_lengths"] = length_sequence[:30000]

testset = {}
testset["process_parameters"] = process_parameters[30000:]
testset["chemical_parameters"] = chemical_parameters[30000:]
testset["cooling_sequence"] = cooling_sequence[30000:]
testset["output_parameters"] = output_variables[30000:]
testset["sequence_lengths"] = length_sequence[30000:]

io.savemat("../Data/train_aim_qlty_sequence.mat", trainset)
io.savemat("../Data/test_aim_qlty_sequence.mat", testset)
# io.savemat("../Data/Cooling_Sequence_data.mat", cooling_seq_dict)
