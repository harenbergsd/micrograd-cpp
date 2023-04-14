#include <vector>
#include <memory>

struct ValueNode
{
    enum class Operation
    {
        None,
        Add,
        Subtract,
        Multiply,
        Tanh,
        Sigmoid
    };

    ValueNode(double value);
    ValueNode(double value, Operation op, ValueNode *parent1, ValueNode *parent2 = nullptr);
    void calcGradientAtParents(); // use chain rule to calculate the gradient on the inner variables

    ValueNode &operator+(ValueNode &rhs); // member function because did not see a way to surface via pybind otherwise
    ValueNode &operator-(ValueNode &rhs); // member function because did not see a way to surface via pybind otherwise
    ValueNode &operator*(ValueNode &rhs); // member function because did not see a way to surface via pybind otherwise
    ValueNode &tanh();                    // member function for consistency with above
    ValueNode &sigmoid();                 // member function for consistency with above

    double value = 0;
    double gradient = 0;
    Operation operation = Operation::None;
    ValueNode *parent1 = nullptr;
    ValueNode *parent2 = nullptr;
};

std::vector<ValueNode *> getTopologicalOrdering(ValueNode &val);
void backprop(ValueNode &val);