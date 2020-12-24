root2hdf5
  source environment: source "/afs/ihep.ac.cn/users/y/yuansc/envs/setup_conda.sh"
  generate h5 files for signal and noise: python dataset/root2hdf5_new_all.py
  
then devide dataset into train and val
  your dataset folder should be like that:
    -sig
      -train
      -val
    -noi
      -train
      -val

train
  train CNN network: sbatch jobs/classify_pos_and_pos+c14_new.sh

