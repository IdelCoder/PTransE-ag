all: ag test pos
ag: Train_PTransE-ag.cpp
	g++ -std=c++0x Train_PTransE-ag.cpp -o Train_PTransE-ag -O2
test: Test_TransE_path.cpp
	g++ -std=c++0x Test_TransE_path.cpp -o Test_TransE_path -O2
pos: Train_PTransE-pos.cpp
	g++ -std=c++0x Train_PTransE-pos.cpp -o Train_PTransE-pos -O2
