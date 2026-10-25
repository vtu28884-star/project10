import kagglehub
import shutil
import glob
import os

print('[+] Downloading dataset...')
path = kagglehub.dataset_download('nasirayub2/human-vital-sign-dataset')
print('[+] Downloaded to cache:', path)

csv_files = glob.glob(os.path.join(path, '*.csv'))
if csv_files:
    shutil.copy(csv_files[0], 'vital_signs_dataset.csv')
    print('[+] Copied CSV file to current folder as vital_signs_dataset.csv')
else:
    print('[!] No CSV file found in downloaded path.')
