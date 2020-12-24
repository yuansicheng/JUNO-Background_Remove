root2hdf5  
  &emsp;source environment: source "/afs/ihep.ac.cn/users/y/yuansc/envs/setup_conda.sh"  
  &emsp;generate h5 files for signal and noise: python dataset/root2hdf5_new_all.py  
      
then devide dataset into train and val  
  &emsp;your dataset folder should be like that:  
    &emsp;&emsp;-sig  
      &emsp;&emsp;&emsp;-train  
      &emsp;&emsp;&emsp;-val  
    &emsp;&emsp;-noi  
      &emsp;&emsp;&emsp;-train  
      &emsp;&emsp;&emsp;-val  
    
train  
  &emsp;train CNN network: sbatch jobs/classify_pos_and_pos+c14_new.sh  

