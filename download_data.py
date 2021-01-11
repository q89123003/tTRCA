import urllib.request
import shutil

# Download the file from `url` and save it locally under `file_name`:

SSVEP_Data = True
ErRP_Data = False

if SSVEP_Data:
    print('SSVEP Data: 35 subjects to download.')
    for sId in range(1, 36):
        print(f'Downloading data of subject {sId}...')
        url = f'ftp://sccn.ucsd.edu/pub/ssvep_benchmark_dataset/S{sId}.mat'
        file_name = f'./data/S{sId}.mat'
        with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    print('Finished.')

if ErRP_Data:
    print('SSVEP Data: 6 subjects to download.')
    for sId in range(1, 7):
        for sessId in range(1, 3):
            print(f'Downloading data of subject {sId} (session {sessId})...')
            url = f'http://bnci-horizon-2020.eu/database/data-sets/013-2015/Subject0{sId}_s{sessId}.mat'
            file_name = f'./data/S{sId}-s{sessId}.mat'
            with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

    print('Finished.')
