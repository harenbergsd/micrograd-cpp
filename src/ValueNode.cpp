#include <cmath>
#include <unordered_set>
#include <functional>
#include <algorithm>

#include "ValueNode.hpp"

ValueNode::ValueNode(double value)
{
    this->value = value;
}
ValueNode::ValueNode(double value, Operation op, ValueNode *parent1, ValueNode *parent2)
{
    this->value = value;
    this->operation = op;
    this->parent1 = parent1;
    this->parent2 = parent2;
}

/*
 * The gradient is based on the chain rule.
 * ValueNode.gradient stores the partial derivative of the final function, call it L, with respect to this ValueNode.
 * The gradient for a parent (p) of the child node (c) is calculated via chain rule: dL/dp = dL/dc * dc/dp.
 * In terms of our ValueNode object, that is p.gradient = this->gradient * <local derivative that depends on the operation>
 */
void ValueNode::calcGradientAtParents()
{
    switch (operation)
    {
    case Operation::Add:
    {
        parent1->gradient += gradient;
        parent2->gradient += gradient;
        break;
    }
    case Operation::Subtract:
    {
        parent1->gradient += gradient;
        parent2->gradient -= gradient;
        break;
    }
    case Operation::Multiply:
    {
        parent1->gradient += gradient * parent2->value;
        parent2->gradient += gradient * parent1->value;
        break;
    }
    case Operation::Tanh:
    {
        parent1->gradient += gradient * (1 - std::pow(value, 2));
        break;
    }
    case Operation::Sigmoid:
    {
        parent1->gradient += gradient * value * (1 - value);
        break;
    }
    case Operation::None:
    default:
        break;
    }
}

ValueNode &ValueNode::operator+(ValueNode &rhs)
{
    ValueNode *v = new ValueNode(value + rhs.value, ValueNode::Operation::Add, this, &rhs);
    return *v;
}
ValueNode &ValueNode::operator-(ValueNode &rhs)
{
    ValueNode *v = new ValueNode(value - rhs.value, ValueNode::Operation::Subtract, this, &rhs);
    return *v;
}
ValueNode &ValueNode::operator*(ValueNode &rhs)
{
    ValueNode *v = new ValueNode(value * rhs.value, ValueNode::Operation::Multiply, this, &rhs);
    return *v;
}
ValueNode &ValueNode::tanh()
{
    ValueNode *v = new ValueNode(std::tanh(value), ValueNode::Operation::Tanh, this);
    return *v;
}
ValueNode &ValueNode::sigmoid()
{
    ValueNode *v = new ValueNode(1.0 / (1 + std::exp(-1 * value)), ValueNode::Operation::Sigmoid, this);
    return *v;
}

std::vector<ValueNode *> getTopologicalOrdering(ValueNode &val)
{
    std::vector<ValueNode *> ordered;
    std::unordered_set<ValueNode *> visitedNodes;

    std::function<void(ValueNode *)> toposort = [&](ValueNode *child)
    {
        if (visitedNodes.find(child) == visitedNodes.end())
        {
            visitedNodes.insert(child);
            if (child->parent1)
                toposort(child->parent1);
            if (child->parent2)
                toposort(child->parent2);
            ordered.push_back(child);
        }
    };

    toposort(&val);
    std::reverse(ordered.begin(), ordered.end());

    return ordered;
}

void backprop(ValueNode &val)
{
    std::vector<ValueNode *> nodes = getTopologicalOrdering(val);

    for (ValueNode *node : nodes)
        node->gradient = 0;

    val.gradient = 1;
    for (ValueNode *node : nodes)
        node->calcGradientAtParents();
}
