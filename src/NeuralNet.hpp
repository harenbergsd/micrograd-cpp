#include <functional>
#include "ValueNode.hpp"

struct Neuron
{
    Neuron(size_t numInputs);
    ValueNode *operator()(std::vector<ValueNode *> &x);

    size_t numInputs;
    std::vector<ValueNode> weights;
    ValueNode bias;
    std::function<ValueNode &(ValueNode *)> activation = &ValueNode::tanh;
};

struct Layer
{
    Layer(size_t numInputsPerNeuron, size_t numNeurons);
    std::vector<ValueNode *> operator()(std::vector<ValueNode *> &x);

    std::vector<Neuron> neurons;
};

struct MLP
{
    MLP(size_t numInputsPerNeuron, std::vector<size_t> numNeurons, std::string const &outActivationName);

    void resetWeights();
    void cleanupNonRootNodes(ValueNode *nodePtr);

    std::vector<ValueNode *> forwardPass(std::vector<ValueNode *> &xin);
    void gradientDescentStep(double stepSize);

    std::vector<double> fit(std::vector<std::vector<double>> const &X, std::vector<std::vector<double>> const &Y, size_t numSteps, double stepSize);
    std::vector<double> predict(std::vector<double> const &X);

    std::vector<Layer> layers;
};

ValueNode &l2loss(std::vector<ValueNode *> &x, std::vector<ValueNode *> &y);
