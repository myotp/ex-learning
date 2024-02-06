defmodule ExLearning.Core.ForwardPropagation do
  import Nx.Defn

  alias ExLearning.Core.Util

  defn sigmoid(z) do
    1 / (1 + Nx.exp(-z))
  end

  defn softmax(logits) do
    exponentials = Nx.exp(logits)
    exponentials / Nx.sum(Nx.exp(logits), axes: [1], keep_axes: true)
  end

  defn cross_entropy_loss(y, y_hat) do
    y_shape_0 = Nx.axis_size(y, 0)
    only_term = y * Nx.log(y_hat)
    -Nx.sum(only_term) / y_shape_0
  end

  defn prepend_bias(x) do
    Util.prepend_column(x, 1)
  end

  defn classify(x, w1, w2) do
    y_hat = forward(x, w1, w2)
    Nx.argmax(y_hat, axis: 1)
  end

  defn forward(x, w1, w2) do
    h = sigmoid(Nx.dot(prepend_bias(x), w1))
    y_hat = softmax(Nx.dot(prepend_bias(h), w2))
    y_hat
  end

  def report(iter, x_train, y_train, x_test, y_test, w1, w2) do
    y_hat = forward(x_train, w1, w2)
    training_loss = cross_entropy_loss(y_train, y_hat) |> Nx.to_number()
    classifications = classify(x_test, w1, w2)
    accuracy = Nx.mean(Nx.equal(classifications, y_test)) |> Nx.to_number()
    IO.puts("Iteration: #{iter}, loss: #{training_loss}, Accuracy: #{accuracy}")
  end
end
