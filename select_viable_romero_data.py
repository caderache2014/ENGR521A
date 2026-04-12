import pickle
import pandas as pd



#shot_output = open("romero_viable_shot_ids.txt", "w")


#should contain 67 columns
cmod_df = pd.read_pickle("cmod_rd.pkl")

#selection of columns relevant for replicating Romero2010 and WGR2023
# 2 shot
# 3 time
# 6 dip_dt
# 9 li 
# 14 kappa
# 30 v_loop_efit
# 35 dli_dt
# 36 ip
# 40 v_loop

romero_values_df = cmod_df[["shot", "time", "dip_dt", "li", "kappa", "v_loop_efit", "dli_dt", "ip","v_loop"]]

romero_shot_vloopefit = cmod_df[["shot", "v_loop_efit"]]

shots_with_NaN_vloopefit = {}
shots_with_viable_vloopefit = {}

romero_viable_vloop_efit_df = romero_values_df[romero_values_df["v_loop_efit"].notna()]
#romero_columns_headings = romero_values_df.columns.tolist()

print("All headers of Allen Wang's CMod dataset")
col_idx = 1

'''
for col_name in romero_columns_headings:
	column_output.write(f'Column {col_idx}: {col_name}\n')
	col_idx = col_idx + 1

column_output.close()


print('First few rows of romero_viable_vloop_efit_df')
print(romero_viable_vloop_efit_df.head())
'''
print('All unique shot ids with viable v_loop_efit values')
viable_shot_ids = romero_viable_vloop_efit_df.shot.unique().tolist()

shot_count = len(viable_shot_ids)

print(f'total number of shots {shot_count}')



for shot_id in viable_shot_ids:

   #current_output_file = open(f'romero_data_shot_{shot_id}.txt', 'w')

   current_shot_subset = romero_viable_vloop_efit_df[romero_viable_vloop_efit_df['shot'] == shot_id]
   current_shot_subset.to_csv(f'romero_shot_{shot_id}.csv', index=False)

'''
shot_output = open('romero_viable_shot_ids.txt', 'w')


shot_idx = 1
for shot_id in viable_shot_ids:
	shot_output.write(f'Item {shot_idx}: shot number: {shot_id}\n')
	shot_idx = shot_idx + 1
shot_output.close()



with pd.option_context('display.max_rows', 1000):
	print(romero_viable_vloop_efit_df)






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

