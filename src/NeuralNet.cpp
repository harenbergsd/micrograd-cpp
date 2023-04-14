#include <algorithm>
#include <random>
#include "NeuralNet.hpp"

static std::random_device r;
// static std::default_random_engine e(r());
static std::default_random_engine e(1234);
static std::uniform_real_distribution<double> uniform_dist(-1, 1);

static std::vector<ValueNode *> convertToValueNodeVec(std::vector<double> const &vec)
{
    std::vector<ValueNode *> v;
    v.reserve(vec.size());
    for (double d : vec)
        v.push_back(new ValueNode(d));
    return v;
}

static std::vector<double> convertFromValueNodeVec(std::vector<ValueNode *> const &vec)
{
    std::vector<double> v;
    v.reserve(vec.size());
    for (ValueNode *val : vec)
        v.push_back(val->value);
    return v;
}

static void freeValueNodes(std::vector<ValueNode *> &vec)
{
    for (ValueNode *v : vec)
        delete v;
}

static void freeValueNodes(std::vector<std::vector<ValueNode *>> &vec)
{
    for (std::vector<ValueNode *> v : vec)
        freeValueNodes(v);
}

Neuron::Neuron(size_t numInputs) : numInputs(numInputs), bias(uniform_dist(e))
{
    weights.reserve(numInputs);
    for (size_t i = 0; i < numInputs; i++)
    {
        weights.emplace_back(uniform_dist(e));
    }
}
ValueNode *Neuron::operator()(std::vector<ValueNode *> &x)
{
    ValueNode *currVal = &bias;
    for (size_t i = 0; i < weights.size(); i++)
    {
        ValueNode &wx = (weights[i]) * (*x[i]);
        ValueNode &newVal = (*currVal) + wx;
        currVal = &newVal;
    }

    ValueNode &finalVal = activation(currVal);

    return &finalVal;
}

Layer::Layer(size_t numInputsPerNeuron, size_t numNeurons)
{
    neurons = std::vector<Neuron>();
    neurons.reserve(numNeurons);
    for (size_t i = 0; i < numNeurons; i++)
        neurons.emplace_back(numInputsPerNeuron);
}
std::vector<ValueNode *> Layer::operator()(std::vector<ValueNode *> &x)
{
    std::vector<ValueNode *> outputs;
    outputs.reserve(neurons.size());
    for (Neuron &neuron : neurons)
        outputs.push_back(neuron(x));
    return outputs;
}

MLP::MLP(size_t numInputsPerNeuron, std::vector<size_t> numNeurons, std::string const &outActivationName)
{
    size_t nLayers = numNeurons.size();

    layers.reserve(nLayers);
    layers.emplace_back(numInputsPerNeuron, numNeurons[0]);
    for (size_t i = 0; i < nLayers - 1; i++)
        layers.emplace_back(numNeurons[i], numNeurons[i + 1]);

    std::function<ValueNode &(ValueNode *)> outActivationFunction = &ValueNode::tanh;
    if (outActivationName == "sigmoid")
        outActivationFunction = &ValueNode::sigmoid;

    for (Neuron &n : layers.back().neurons)
        n.activation = outActivationFunction;
}
std::vector<ValueNode *> MLP::forwardPass(std::vector<ValueNode *> &xin)
{
    std::vector<ValueNode *> x = xin;
    for (Layer &layer : layers)
        x = layer(x);
    return x;
}

void MLP::gradientDescentStep(double stepSize)
{
    for (Layer &l : this->layers)
    {
        for (Neuron &n : l.neurons)
        {
            n.bias.value += -1 * stepSize * n.bias.gradient;
            for (ValueNode &v : n.weights)
            {
                v.value += -1 * stepSize * v.gradient;
            }
        }
    }
}

void MLP::resetWeights()
{
    for (Layer &l : this->layers)
    {
        for (Neuron &n : l.neurons)
        {
            n.bias.value = uniform_dist(e);
            for (ValueNode &v : n.weights)
            {
                v.value = uniform_dist(e);
            }
        }
    }
}

void MLP::cleanupNonRootNodes(ValueNode *nodePtr)
{
    std::vector<ValueNode *> nodes = getTopologicalOrdering(*nodePtr);
    for (ValueNode *node : nodes)
    {
        if (node->parent1 || node->parent2)
        {
            delete node;
        }
    }
}

std::vector<double> MLP::fit(std::vector<std::vector<double>> const &Xin, std::vector<std::vector<double>> const &Yin, size_t numSteps, double stepSize)
{
    resetWeights();

    std::vector<std::vector<ValueNode *>> Xs;
    std::vector<std::vector<ValueNode *>> Ys;
    Xs.reserve(Xin.size());
    Ys.reserve(Yin.size());
    std::transform(Xin.begin(), Xin.end(), std::back_inserter(Xs), convertToValueNodeVec);
    std::transform(Yin.begin(), Yin.end(), std::back_inserter(Ys), convertToValueNodeVec);

    std::vector<double> losses;
    for (size_t i = 0; i < numSteps; i++)
    {
        std::vector<std::vector<ValueNode *>> ypred;
        for (std::vector<ValueNode *> xin : Xs)
            ypred.push_back(forwardPass(xin));

        ValueNode *loss = new ValueNode(0);
        for (size_t i = 0; i < ypred.size(); i++)
        {
            ValueNode &newLoss = *loss + l2loss(ypred[i], Ys[i]);
            loss = &newLoss;
        }

        backprop(*loss);

        gradientDescentStep(stepSize);

        losses.push_back(loss->value);
        cleanupNonRootNodes(loss);
    };

    freeValueNodes(Xs);
    freeValueNodes(Ys);

    return losses;
}

std::vector<double> MLP::predict(std::vector<double> const &xin)
{
    std::vector<ValueNode *> x = convertToValueNodeVec(xin);
    std::vector<ValueNode *> ypred = forwardPass(x);
    std::vector<double> y = convertFromValueNodeVec(ypred);

    freeValueNodes(x);
    freeValueNodes(ypred);
    return y;
}

ValueNode &l2loss(std::vector<ValueNode *> &x, std::vector<ValueNode *> &y)
{
    ValueNode *l2 = new ValueNode(0);
    for (size_t i = 0; i < x.size(); i++)
    {
        ValueNode &xi = *x[i];
        ValueNode &yi = *y[i];
        ValueNode &si = ((xi - yi) * (xi - yi)) + (*l2);
        l2 = &si;
    }
    return *l2;
}