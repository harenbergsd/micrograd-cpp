g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) src/ValueNode.cpp src/NeuralNet.cpp src/PythonInterface.cpp -o micrograd$(python3-config --extension-suffix)