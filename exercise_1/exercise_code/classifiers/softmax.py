"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier

# class LinearClassifier():

#     def __init__(self, a):
#         self.a = a


def cross_entropoy_loss_naive(W, X, y, reg):

    #return cross_entropoy_loss_vectorized(W, X, y, reg)
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]

    reg_kind = "L1"

    outputs = np.zeros((C,))

    for num_sample, sample in enumerate(X):

        for num_class in range(C):
            incoming = 0.

            for var in range(D):
                incoming += sample[var]*W[var, num_class]

            outputs[num_class] = np.exp(incoming)

        outputs = outputs/sum(outputs)

        # print(outputs)

        loss += - np.log(outputs[y[num_sample]])

        loss += 0.5*reg*np.sum(W**2)

        for num_class in range(C):
            dW[:,num_class] += np.reshape(outputs[num_class] * sample, dW[:, num_class].shape) + reg * W[:, num_class]

            if num_class == y[num_sample]:
                dW[:,num_class] -= sample.reshape(dW[:, num_class].shape)

    dW = dW/N
    loss = loss/N



    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    numeric_stability = 1e-9

    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]

    outputs = np.matmul(X, W)

    stable_outputs = outputs - np.max(outputs)

    stable_outputs_softmax = np.exp(stable_outputs)

    stable_outputs_softmax = np.maximum(stable_outputs_softmax, np.ones(outputs.shape)*numeric_stability)

    norm_outputs = stable_outputs_softmax/stable_outputs_softmax.sum(axis=1)[:, np.newaxis]


    loss = - np.sum(np.log(norm_outputs[range(N), y]))

    loss /= N

    loss += 0.5 * reg * np.sum(W**2)

    norm_outputs[range(N), y] -= 1

    dW = np.matmul(X.T, norm_outputs)

    dW /= N

    dW += reg*W

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = np.linspace(1e-6, 1e-5, 10).tolist()
    print(learning_rates)
    regularization_strengths = [1e-4, 2.5e-4, 5e-4, 1e-5, 2.5e-5, 5e-5] # , 2.5e4, 5e4

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    for learning_rate in learning_rates:
        for regularization_strength in regularization_strengths:
            classifier = SoftmaxClassifier()

            loss_history = classifier.train(X_train, y_train, learning_rate=learning_rate, reg=regularization_strength, num_iters=1000)

            y_train_pred = classifier.predict(X_train)
            training_accuracy = np.mean(y_train == y_train_pred)

            y_val_pred = classifier.predict(X_val)
            validation_accuracy = np.mean(y_val == y_val_pred)

            if validation_accuracy > best_val:
                best_val = validation_accuracy
                best_softmax = classifier

            results[(learning_rate, regularization_strength)] = (training_accuracy, validation_accuracy)

            all_classifiers.append(classifier)


    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers

if __name__ == "__main__":

    W = np.ones((3,2))

    X = np.array([[1, 2, 3], [4, 5, 6]])

    y = np.array([0,1])

    gamma = 0.00001

    x = cross_entropoy_loss_naive(W,X,y,gamma)
