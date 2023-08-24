# Sparsity-Brings-Vunerabilities
This is the code repository of **Sparsity Brings Vulnerabilities: Exploring New Metrics in Backdoor Attacks** Usenix Security 2023

## Environments
We suggest to creat a virtual environments for the replication``conda create -n sbv python=3.9`` and ``conda activate sbv``

### prepare the environment
``pip install -r requirements.txt``

### Install ember
Install EMBER:
``cd ember/ && python setup.py install`` or ``pip install git+https://github.com/elastic/ember.git``

### Prepare the dataset
The extracted PDF dataset is placed in `datasets/pdf/dataset.csv`, you can directly use it. If you still need the original PDF files, please acquire them from [Contagio](https://contagiodump.blogspot.com/2013/03/16800-clean-and-11960- malicious-files.html)

For DREBIN, we placed two files of sha256 (`benign.txt` and `malware.txt`) in `datasets/drebin/`, please download them from Androzoo.

The code will automatically download EMBER datasets.

### Create base models
```
#EMBER
python train_model.py -m nn -d ember --save_dir models/ember/ --save_file base_nn
python train_model.py -m lightgbm -d ember --save_dir models/ember/ --save_file base_lightgbm.pkl

#DREBIN
python train_model.py -m nn -d drebin --save_dir models/drebin/ --save_file base_nn
python train_model.py -m linearsvm -d drebin --save_dir models/drebin/ --save_file base_svm.pkl

#PDF
python train_model.py -m nn -d pdf --save_dir models/pdf/ --save_file base_nn
python train_model.py -m rf -d pdf --save_dir models/pdf/ --save_file base_rf.pkl
```
### Run attacks
```
python backdoor_attack.py -c configs/unrestricted_table1.json
```
-------------------------------------------------------------------------------------------
Since the original code was written on jupyter, the code was quite scattered. We want to refactor it, which will take some time. We have released some code of the unrestricted attack, and others will coming soon.

