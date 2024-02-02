import papermill as pm
from time import time, gmtime

# This file, all other imported .py files, and the blank notebooks should be visible in the same folder.
# Otherwise, some paths will have to be changed in various places.

# Run file with this command:
# nohup python run_multi_models.py > multi_model_run.log &


MIN_NB_NUMB = 1
MAX_NB_NUMB = 6

print(f'Training {MAX_NB_NUMB - MIN_NB_NUMB} models.')

start_time = time()

for nb_val in range(MIN_NB_NUMB, MAX_NB_NUMB+1):
   pm.execute_notebook(
      f'./CNN_Model_{nb_val}.ipynb',
      f'./trained_nbs/CNN_Model_{nb_val}_trained.ipynb'
   )
total_time = time() - start_time
conv_time = gmtime(total_time)
print(f'Training finished in: {conv_time.tm_hour} hours, {conv_time.tm_min} minutes, {conv_time.tm_sec} seconds')