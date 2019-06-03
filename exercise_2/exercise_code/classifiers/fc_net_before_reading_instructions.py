import numpy as np

from exercise_code.layers import *
from exercise_code.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    
    The architecure should be affine - relu - affine - softmax.
  
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
  
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.
    
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################

        #self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['W1'] = np.random.normal(0., weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        #self.params['W2'] = weight_scale * np.random.randn(hidden_dim, output_dim)
        self.params['W2'] = np.random.normal(0., weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
    
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
    
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
    
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        H, C = W2.shape

        numeric_stability = 1e-9

        first_layer_input = np.matmul(X, W1) + b1

        first_layer_output = np.maximum(first_layer_input, 0)

        second_layer_input = np.matmul(first_layer_output, W2) + b2

        scores = second_layer_input

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        stable_outputs = second_layer_input - np.max(second_layer_input)

        stable_outputs_softmax = np.exp(stable_outputs)

        stable_outputs_softmax = np.maximum(stable_outputs_softmax, np.ones(stable_outputs_softmax.shape)*numeric_stability)

        scores = stable_outputs_softmax/stable_outputs_softmax.sum(axis=1)[:, np.newaxis]

        loss = -1 * np.sum(np.log(scores[range(N), y]))

        loss /= N

        loss += 0.5 * self.reg * np.sum(W1**2)
        loss += 0.5 * self.reg * np.sum(W2**2)

        scores[range(N), y] -= 1

        dW2 = np.matmul(first_layer_output.T, scores)

        dW2 /= N

        dW2 += self.reg*W2

        db2 = np.sum(scores, axis=0)

        db2 /= N

        middle_layer_derivative = np.matmul(scores, W2.T)

        ReLu_derivative = np.where(first_layer_input>0, np.ones(first_layer_input.shape), np.zeros(first_layer_input.shape))

        dW1 = np.matmul(X.T, middle_layer_derivative*ReLu_derivative)

        dW1 /= N

        dW1 += self.reg*W1

        db1 = np.sum(middle_layer_derivative*ReLu_derivative, axis=0)

        db1 /= N

        grads["W2"] = dW2
        grads["W1"] = dW1
        grads["b2"] = db2
        grads["b1"] = db1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be
    
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    
    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
    
    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.
        
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################

        self.params['W1'] = np.random.normal(0., weight_scale, (input_dim, hidden_dims[0]))
        self.params['b1'] = np.zeros(hidden_dims[0])

        for i, dim in enumerate(hidden_dims):

            w_name = 'W%s'%(i+2)
            b_name = 'b%s'%(i+2)

            if i+1 == len(hidden_dims):
                self.params[w_name] = np.random.normal(0., weight_scale, (dim, num_classes))
                self.params[b_name] = np.zeros(num_classes)
                continue

            self.params[w_name] = np.random.normal(0., weight_scale, (dim, hidden_dims[i+1]))
            self.params[b_name] = np.zeros(hidden_dims[i+1])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
    
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        inputs = []
        outputs = []

        first_layer_input = np.matmul(X, self.params['W1']) + self.params['b1']
        first_layer_output = np.maximum(first_layer_input, 0.)

        inputs.append(first_layer_input)
        outputs.append(first_layer_output)

        for i in range(1, self.num_layers):
            weights_name = 'W%s'%(i+1)
            bias_name = 'b%s'%(i+1)

            weights = self.params[weights_name]
            bias = self.params[bias_name]

            new_input = np.matmul(outputs[-1], weights) + bias_name
            new_output = np.maximum(new_input, 0.)

            inputs.append(new_input)
            outputs.append(new_output)

        scores = inputs[-1]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        N, D = X.shape

        layer_derivatives = []

        numeric_stability = 1e-8

        stable_outputs = scores - np.max(scores)

        stable_outputs_softmax = np.exp(stable_outputs)

        stable_outputs_softmax = np.maximum(stable_outputs_softmax, np.ones(stable_outputs_softmax.shape)*numeric_stability)

        softmax_result = stable_outputs_softmax/stable_outputs_softmax.sum(axis=1)[:, np.newaxis]

        loss = -1 * np.sum(np.log(softmax_result[range(N), y]))

        loss /= N

        for i in range(self.num_layers):

            weights_name = 'W%s'%(i+1)

            loss += 0.5 * self.reg * np.sum(self.params[weights_name]**2)



        softmax_result[range(N), y] -= 1

        dW_last = np.matmul(first_layer_output.T, softmax_result)

        dW_last /= N

        dW_last += self.reg*W2

        db_last = np.sum(softmax_result, axis=0)

        db_last /= N

        for i in range(self.num_layers):
            weights_name = 'W%s'%(self.num_layers-i)
            bias_name = 'b%s'%(self.num_layers-i)

            if i == 0:
                grads[weights_name] = dW_last
                grads[bias_name] = db_last

                ReLu_derivative = np.where(inputs[-2]>0, np.ones(inputs[-2].shape), np.zeros(inputs[-2].shape))

                last_hidden_layer_derivative = np.matmul(softmax_result, self.params[weights_name].T)

                layer_derivatives.append(last_hidden_layer_derivative*ReLu_derivative)

                continue

            if i+1 == self.num_layers:
                dW = np.matmul(X.T, layer_derivatives[0])
                dW /= N
                db = np.sum(layer_derivatives[0], axis=0)
                db /= N

                grads[weights_name] = dW
                grads[bias_name] = db

                continue

            dW = np.matmul(outputs[-i-2].T, layer_derivatives[0])
            dW /= N
            db = np.sum(layer_derivatives[0], axis=0)
            db /= N

            grads[weights_name] = dW
            grads[bias_name] = db

            next_layer_derivative = np.matmul(layer_derivatives[0], self.params[weights_name].T)
            ReLu_derivative = np.where(inputs[-i-2]>0, np.ones(inputs[-i-2].shape), np.zeros(inputs[-i-2].shape))

            layer_derivatives.insert(0, next_layer_derivative*ReLu_derivative)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
