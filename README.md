# Artificial Neural Networks, Deep Learning Churn Modeling

## Overview of Deep Learning: 
Introduction
- Neural networks & deep learning been around for some time. 60s and 70s invented, 80s caught wind with tons of research and thought it would solve all the world's problem...and then slowly died off in the next decade.
- Was it because they weren't good enough? Or was there another reason? Technology at the time was not at the standard to facilate neural networks. Need lots of data and processing power.
- Storage 
  - 1956 Hard drive (5 MB - Massive closet size, 2.5k to rent a month)
  - 1980 Hard drive (10 MB $3500) 
  - 2017 Hard drive (256 GB $149)
  - capacity of whatever was trending, 1956 -> 1980 = 2x, 1980 -> 2017 25,600x 
  - not a linear trend, exponential trend - logarithmic scale
  - DNA storage (future, $7k to synthesize 2MB of data, $2k to read it now)
  - 1kg of weight needed of DNA to store all of the world's data - 1 billion terabytes in one gram of DNA
  - Deep learning is picking up now since we have enough data now
  - Processing power - Moore's law - will lead to the Singularity (2023 Surpasses brainpower in humans, 2045 all the humans together)

What is Deep Learning?
- Geoffrey Hinton is the father of Deep Learning (research in 80s) Working at Google
- Idea behind deep learning is to look at the human brain, neuroscience, mimick how the human brain and recreate it. Humans brains are one of the most powerful learning tools in the world.
- Neurons - smeared onto glass - body, branches, tails, nucleus, etc. - 100 billion neurons in the human brain, each neuron is connected to a thousand or so of its neighbor. Cerebellum - motor/balance
- How to recreate this? Artificial Neural Networks - neurons/nodes
  - Input layer: value 1, Input value 2, Input value 3 
  - Output layer: value: (Fradulant transcations, etc)
  - Hidden layer: brain has so many neurons (ears, sense, eyes) going through billions of neurons before to output. 
  - Input later -> Hidden layer -> Output Layer
  - So why is this called Deep Learning? 
  - Shallow learning
  - Not just one hidden layer, multiple hidden layers, connect everything and interconnect, that's why we call it deep learning


Tooling
- Anaconda - prepackaged solutions, IDEs, most common packages, convenient package solution.
- Spyder (IDE) of choice z

## Artificial Neural Network (ANN) Intuition + Background
- Overview:
  - Neuron - how the human brain works
  - Activiation function - which one of them commonly used in neural network and which layer to use them
  - How do Neural Networks work? (example) - working first to see what we're aiming for, simplified version (real estate)
  - How do Neural Networks learn?
  - Gradient Descent
  - Stochastic Gradient Descent 
  - Backpropagation
- Neuron:
  - Neuron basic building blocks of ANNs
  - How can we recreate a neuron in a machine to mimick how the human brain work in the hopes of creating an infrastructure to learn. The human brain is one of the most powerful learning tool in the planet. 
  - First step of ANN is to recreate a Neuron
  - Neurons with lots of branches, thread coming out the bottom (1800s)
  - Neuron (body, dendrites - branches, axon - tail) - neuron by the itself is useless (like ants) but with lots of ants you can build a colony. With lots of neurons they can work together.
  - Axon is the transmitter, dendrites is the receptors, axon connects to the dendrites of the next neuron doesn't really touch - neurotransmitter molecules with receptor and synapse. How signal being passed synapse
  - Synapses is where signal is being passed
  - How we're going to represent neurons in machines? 
    - Neuron gets lots of input signals (X1, X2, Xm) and an output signal
    - Input layer think of it as analogy of the human brain (senses, sight, smell), your brain is only getting electrical impulses from our organs, lives in a dark black box and making sense of the world through input
    - Input layer passed through synapses
    - Neuron (hidden layer)
  - Input layers:
    - Independent variables (rows in DB - age, money, transporation, of a person)
    - Standardize the variables, sometimes want to normalize them -> want all of these variables to be similar and range of values, weights added up, neural networks if they're all the same is easier
    - Reading to learn more about standarization or normalization of variables: Yann LeCun et al., 1998, Efficient BackProp - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf 
    - Output value
      - Continuous (price)
      - Binary (will exit yes/no)
      - Categorical (Output will be several y1, y2, y3)
    - Single Observation on left and right (input) and (right) rows
    - Linear regression or multi-variate linear regression (simplifying) -> one row correlated to that row
    - Synapses 
      - Assigned weights (w1, w2, Wm) 
      - Weights are crucial to ANN, how neural networks learn, adjusting weights what signal is important and what is not important to a neuron, what strength and what extent signals get passed along, weights are crucial and they're the ones that get adjusted. 
      - Weights in all the synapses, training in ANN => Gradient descent and back propogation come into play
    - Neuron
      - signals go into neuron
      - What happens inside the neuron?
        - 1) Weighted sum of all input values
        - 2) Applies an Activation Function - assigned to the layer (neuron), applied to weighted sum, neuron knows to whether to pass on signal on or not
        - 3) Passes that signal to the next neuron down the line
- Activation Function
  - 
