# PTransE-ag
PTransE-ag (path-based TransE with aggregation) is a method for knowledge graph embedding.
**Note that our work is based on PTransE. Therefore, the code is modified based on PTransE (https://github.com/thunlp/KB2E).
Here I only supply dataset FB15K. You can access https://github.com/thunlp/KB2E for more datasets.**

## training
You need to run *PCRA.py* firstly to generate required files. Then modify the files' paths in *Test_Train_path.cpp* and *Train_PTransE-ag.cpp*.
Parameter *beta* is required for running trainning. For our paper's result, you can run **./Train_PTransE-ag 0.5**.

## testing
According to the *beta* passed in training, run **./Test_PTransE-ag beta** for evaluating.
