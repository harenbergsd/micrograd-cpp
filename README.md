# micrograd-cpp

This is a (not yet very good) implementation of a small, reverse-mode automatic differentiation engine and an accompanying neural network written in c++. These both have python bindings via pybind11, but the interface is pretty limited.

This was developed for educational purposes and was created when following kparthy's [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero). You can see his original micrograd repo at [https://github.com/karpathy/micrograd](https://github.com/karpathy/micrograd).

## Usage

You can build by running the build script:
```bash
./build.sh
```

You can see an example of building a neural network classifier in [demo.ipynb](demo.ipynb).

You can also build and calculate gradients for simple algebraic equations in python:
```python
from micrograd import ValueNode, backprop

# Build simple algebraic expression
a = ValueNode(1.0)
b = ValueNode(2.0)
c = ValueNode(3.0)
d = a * b
e = d + c
f = e * e

# Backprop gradient for function f = e^2 = (a*b + c)^2 = a^2b^2 + 2cab + c^2
backprop(f)

print(a)  # prints (val=1.000000, grad=20.000000), which is df / da
print(e)  # prints (val=5.000000, grad=10.000000), which is df / de

```

## Notes

* The neural network in `demo.ipynb` had pretty poor testing accuracy, probably due to overfitting and a small dataset. We have no regularization yet. Also, softmax would be a better activation function for this application.
* This c++ code is not great. There are many aspects that could be improved, and things that could be modernized, like removing the use of raw pointers. However, some things are non-trivial when having to consider how a python program can interface with your objects.
* Surfacing an interface in python led to complications because you have to decide [who is responsible for handling memory](https://pybind11.readthedocs.io/en/stable/advanced/functions.html). In the current design, ValueNodes allocate other ValueNodes during operations, and c++ or python must release that memory, but not both. I think if I were to do it again, I would not let ValueNode, be used as a standalone object, but rather some Expression or Computation class that holds all the ValueNodes that are spawned during a computation.