#£¡/bin/bash

source ~/envs/setup_conda.sh


INPUT=/cefs/higgs/wxfang/JUNO/noise_remove/positron_skim
OUTPUT=positron_skim_dataset.h5

python ../root2hdf5_new.py --input $INPUT/"*.root" --output $OUTPUT

