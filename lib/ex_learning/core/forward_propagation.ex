defmodule ExLearning.Core.ForwardPropagation do
  import Nx.Defn

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

  def prepend_bias(x) do
    prepend_column(x, 1)
  end

  def prepend_column(t, num) do
    rows = Nx.axis_size(t, 0)
    dummy_column = Stream.repeatedly(fn -> num end) |> Enum.take(rows)
    tt = Nx.transpose(t)
    t1 = Nx.tensor([dummy_column])

    Nx.concatenate([t1, tt])
    |> Nx.transpose()
  end

  def classify(x, w1, w2) do
    y_hat = forward(x, w1, w2)
    Nx.argmax(y_hat, axis: 1)
  end

  def forward(x, w1, w2) do
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
