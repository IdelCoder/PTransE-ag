all: train test
train: Train_PTransE-ag.cpp
	g++ -std=c++0x Train_PTransE-ag.cpp -o Train_PTransE-ag -O2
test: Test_TransE_path.cpp
	g++ -std=c++0x Test_TransE_path.cpp -o Test_TransE_path -O2
