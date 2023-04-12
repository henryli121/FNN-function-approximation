# FNN-function-approximation
<p>A capstone project from my undergraduate studies. Using feedforward neural networks to approximate the output of simple multivariable functions.</p>

<p>This code is modified from code samples for "Neural Networks and Deep Learning" by Michael Nielsen. In his book, he constructed a feedforward neural network that can identify handwritten digits using the MNIST dataset. Inspired by this, my undergraduate professor, Andrew Leahy, and I studied the mathematics behind neural networks and revised the code to approximate some general functions, such as:</p>

<pre>
f(x<sub>1</sub>, x<sub>2</sub>) = cos(x<sub>1</sub>) * sin(x<sub>2</sub>)
</pre>

<pre>
g(x<sub>1</sub>, x<sub>2</sub>) = (x<sub>1</sub><sup>2</sup> + x<sub>2</sub><sup>2</sup>) / 2
</pre>

<p>More examples of the funtion are in data folder. Moreover, the <code>paper.pdf</code> provides the proofs for the four fundamental equations of the backpropagation in FNN that raised in Nielsen's book, "Neural Networks and Deep Learning". 
<p>The original code repository is here: <a href="https://github.com/mnielsen/neural-networks-and-deep-learning#license" target="_blank">neural-networks-and-deep-learning</a></p>
