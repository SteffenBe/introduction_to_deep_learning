"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        H, C = W2.shape

        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################

        numeric_stability = 1e-9

        #outputs = np.matmul(np.maximum(np.matmul(X, W1) + b1, np.zeros((N, H))), W2) + b2

        first_layer_input = np.matmul(X, W1) + b1

        first_layer_output = np.maximum(first_layer_input, 0)

        second_layer_input = np.matmul(first_layer_output, W2) + b2

        scores = second_layer_input

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################

        stable_outputs = second_layer_input - np.max(second_layer_input)

        stable_outputs_softmax = np.exp(stable_outputs)

        stable_outputs_softmax = np.maximum(stable_outputs_softmax, np.ones(stable_outputs_softmax.shape)*numeric_stability)

        scores = stable_outputs_softmax/stable_outputs_softmax.sum(axis=1)[:, np.newaxis]

        loss = -1 * np.sum(np.log(scores[range(N), y]))

        loss /= N

        #loss += 0.5 * reg * np.sum(np.sum(W1**2) , np.sum(W2**2) , np.sum(b1**2) , np.sum(b2**2))
        loss += 0.5 * reg * np.sum(W1**2)
        loss += 0.5 * reg * np.sum(W2**2)

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################

        scores[range(N), y] -= 1

        dW2 = np.matmul(first_layer_output.T, scores)

        dW2 /= N

        dW2 += reg*W2

        db2 = np.sum(scores, axis=0)

        db2 /= N

        middle_layer_derivative = np.matmul(scores, W2.T)

        ReLu_derivative = np.where(first_layer_input>0, np.ones(first_layer_input.shape), np.zeros(first_layer_input.shape))

        dW1 = np.matmul(X.T, middle_layer_derivative*ReLu_derivative)

        dW1 /= N

        dW1 += reg*W1

        db1 = np.sum(middle_layer_derivative*ReLu_derivative, axis=0)

        db1 /= N

        grads["W2"] = dW2
        grads["W1"] = dW1
        grads["b2"] = db2
        grads["b1"] = db1

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################

            N, D = X.shape

            indices = np.random.choice(N, batch_size)

            X_batch = X[indices, :]
            y_batch = y[indices]

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################

            self.params["W1"] -= learning_rate * grads["W1"]
            self.params["W2"] -= learning_rate * grads["W2"]
            self.params["b1"] -= learning_rate * grads["b1"]
            self.params["b2"] -= learning_rate * grads["b2"]

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        numeric_stability = 1e-9

        first_layer_input = np.matmul(X, W1) + b1

        first_layer_output = np.maximum(first_layer_input, 0)

        second_layer_input = np.matmul(first_layer_output, W2) + b2

        y_pred = np.argmax(second_layer_input, axis=1)

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    best_net = None # store the best model into this 

    ############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above; these visualizations will have significant      #
    # qualitative differences from the ones we saw above for the poorly tuned  #
    # network.                                                                 #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################

    results = {}
    best_acc = -1
    net_info = ""

    learning_rates = [5e-2, 1e-2, 1e-3, 1e-4, 1e-5] # [1e-3, 5e-3, 1e-4, 5e-4, 1e-5]
    learning_rate_decays = [0.97, 0.95, 0.92, 0.87] # [0.95, 0.92]
    regs = [1e-3] # [1e-3, 1e-4, 1e-5]
    num_iters = [2000, 4000] # [100, 200, 500, 2000]
    batch_sizes = [200, 400] # [16, 64, 200]
    hidden_sizes = [130, 400] # [50, 100, 130]

    # best so far: learning rate 1e-4, 0.92, 5e-4

    for learning_rate in learning_rates:
        for learning_rate_decay in learning_rate_decays:
            for reg in regs:
                for num_iter in num_iters:
                    for batch_size in batch_sizes:
                        for hidden_size in hidden_sizes:
                            net = TwoLayerNet(input_size=32 * 32 * 3, hidden_size=hidden_size, output_size=10)

                            stats = net.train(X_train, y_train, X_val, y_val, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
                                reg=reg, num_iters=num_iter, batch_size=batch_size)

                            y_val_pred = net.predict(X_val)
                            validation_accuracy = np.mean(y_val == y_val_pred)

                            output_string = "VALIDATION ACCURACY: %s ; lr = %s; lr_decay = %s, reg = %s, num_iter = %s, batch = %s, hidden_size = %s" \
                                %(validation_accuracy, learning_rate, learning_rate_decay, reg, num_iter, batch_size, hidden_size)

                            print(output_string)

                            if validation_accuracy > best_acc:
                                best_acc = validation_accuracy
                                best_net = net
                                net_info = output_string

    print("------------------------------------------------------------------")
    print("Best result with acc of %s found: \n" %(best_acc))
    print(net_info)


    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net
