#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "NeuralNet.hpp"

PYBIND11_MODULE(micrograd, m)
{
    pybind11::class_<ValueNode>(m, "ValueNode")
        .def(pybind11::init<double>())
        .def("__repr__", [](const ValueNode &v)
             { return "(val=" + std::to_string(v.value) + ", grad=" + std::to_string(v.gradient) + ")"; })
        .def("calcGradientAtParents", &ValueNode::calcGradientAtParents)
        .def("__add__", &ValueNode::operator+)
        .def("__sub__", &ValueNode::operator-)
        .def("__mul__", &ValueNode::operator*)
        .def("tanh", &ValueNode::tanh)
        .def("sigmoid", &ValueNode::sigmoid);

    pybind11::class_<Neuron>(m, "Neuron")
        .def(pybind11::init<size_t>())
        .def("__call__", &Neuron::operator());

    pybind11::class_<MLP>(m, "MLP")
        .def(pybind11::init<size_t, std::vector<size_t>, std::string const &>())
        .def("fit", &MLP::fit)
        .def("predict", &MLP::predict);
    ;

    m.def("backprop", &backprop);
};
