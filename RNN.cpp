#include <iostream>
#include <armadillo>

int inputSize = 10;  // Number of input units
int hiddenSize = 20;  // Number of hidden units
int outputSize = 5;  // Number of output units
int timeSteps = 100;  // Number of time steps

arma::mat Wxh(hiddenSize, inputSize, arma::fill::randn);  // Input to hidden weights
arma::mat Whh(hiddenSize, hiddenSize, arma::fill::randn);  // Hidden to hidden weights
arma::mat Why(outputSize, hiddenSize, arma::fill::randn);  // Hidden to output weights

arma::vec bh(hiddenSize, arma::fill::zeros);  // Hidden bias
arma::vec by(outputSize, arma::fill::zeros);  // Output bias

arma::mat rnnForward(const arma::mat& inputs) {
    int seqLength = inputs.n_cols;
    arma::mat hiddenStates(hiddenSize, seqLength, arma::fill::zeros);
    arma::mat outputs(outputSize, seqLength, arma::fill::zeros);

    arma::vec hPrev(hiddenSize, arma::fill::zeros);

    for (int t = 0; t < seqLength; ++t) {
        arma::vec x = inputs.col(t);
        arma::vec h = arma::tanh(Wxh * x + Whh * hPrev + bh);
        arma::vec y = Why * h + by;

        hiddenStates.col(t) = h;
        outputs.col(t) = y;

        hPrev = h;
    }

    return outputs;
}

int main() {
    // Example usage
    arma::mat inputs(inputSize, timeSteps, arma::fill::randn);

    arma::mat outputs = rnnForward(inputs);

    std::cout << "Output shape: " << outputs.n_rows << " x " << outputs.n_cols << std::endl;

    return 0;
}