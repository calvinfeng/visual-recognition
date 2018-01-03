# Visual Recognition
This is primarily a test bed for my raw convolutional neural network implementation without the aid of tensor flow and pytorch.
It is also a repository for the mathematical derivation of local gradient in each layer.

The math is in the `notebooks` directory of each project.

## How to run
### Virtual environment
First of all, create a virtual environment to manage all your pip dependencies:
```
virtualenv environment
```

And then activate the environment:
```
source environment/bin/activiate
```

And then pip install the requirements:
```
pip install -r requirements.txt
```

Now you are ready to go.

### Run the code
I recommend using jupyter notebook to run my code. Go to the notebooks section of each project, and then run
```
jupyter notebook
```

Your default browser will open up a new tab and it be pretty self-explanatory.

### Run unit tests
Go to each directory and enter
```
nosetests
```

## Neural Networks
There are three directories for neural networks, starting with the `neural_net`, `demo_net` and then `conv_net`. The former
two are test implementations, and the latter is the modularized implementation, i.e. the each layer of the network is
modularized, they can be easily swapped in and out. The `conv_net` has the best unit test coverage. Eventually the modularized
implementation will replace all other code in this repo.
