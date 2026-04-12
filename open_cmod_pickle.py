import pickle
import pandas as pd


column_output = open("cmod_columns.txt", "w")

df = pd.read_pickle("cmod_rd.pkl")
columns_headings = df.columns.tolist()

print("All headers of Allen Wang's CMod dataset")
col_idx = 1
for col_name in columns_headings:
	column_output.write(f'Column {col_idx}: {col_name}\n')
	col_idx = col_idx + 1

column_output.close()


'''
with open('./cmod_rd.pkl', 'rb') as f:

	p = pickle.load(f)
print(p)


#convert original data into serialized 
pickle.dump()  #for file type object
pickle.dumps() #for byte sting object

#convert serialized 'pickle' format to orignal form
pickle.load()
pickle.loads()
'''

