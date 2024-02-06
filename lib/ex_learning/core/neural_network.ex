defmodule ExLearning.Core.NeuralNetwork do
  import Nx.Defn

  defn sigmoid(z) do
    1 / (1 + Nx.exp(-z))
  end

  defn sigmoid_gradient(sigmoid) do
    sigmoid * (1 - sigmoid)
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
    append_column(x, 1)
  end

  def append_column(t, num) do
    rows = Nx.axis_size(t, 0)
    dummy_column = Stream.repeatedly(fn -> num end) |> Enum.take(rows)
    tt = Nx.transpose(t)
    t1 = Nx.tensor([dummy_column])

    Nx.concatenate([t1, tt])
    |> Nx.transpose()
  end

  def classify(x, w1, w2) do
    {y_hat, _} = forward(x, w1, w2)
    Nx.argmax(y_hat, axis: 1)
  end

  def forward(x, w1, w2) do
    h = sigmoid(Nx.dot(prepend_bias(x), w1))
    y_hat = softmax(Nx.dot(prepend_bias(h), w2))
    {y_hat, h}
  end

  def back(x, y, y_hat, w2, h) do
    w2_shape_0 = Nx.axis_size(w2, 0)
    x_shape_0 = Nx.axis_size(x, 0)

    w2_gradient =
      Nx.divide(
        Nx.dot(Nx.transpose(prepend_bias(h)), Nx.subtract(y_hat, y)),
        x_shape_0
      )

    w2_no_first_row = Nx.slice_along_axis(w2, 1, w2_shape_0 - 1, axis: 0)

    w1_gradient =
      Nx.divide(
        Nx.dot(
          Nx.transpose(prepend_bias(x)),
          Nx.multiply(
            Nx.dot(
              Nx.subtract(y_hat, y),
              Nx.transpose(w2_no_first_row)
            ),
            sigmoid_gradient(h)
          )
        ),
        x_shape_0
      )

    {w1_gradient, w2_gradient}
  end

  def report(iter, x_train, y_train, x_test, y_test, w1, w2) do
    {y_hat, _} = forward(x_train, w1, w2)
    training_loss = cross_entropy_loss(y_train, y_hat) |> Nx.to_number()
    classifications = classify(x_test, w1, w2)
    accuracy = Nx.mean(Nx.equal(classifications, y_test)) |> Nx.to_number()
    IO.puts("Iteration: #{iter}, loss: #{training_loss}, Accuracy: #{accuracy}")
  end

  def initialize_weights(n_input_variables, n_hidden_nodes, n_output_classes) do
    w1_rows = n_input_variables + 1
    w2_rows = n_hidden_nodes + 1
    w1_shape = {w1_rows, n_hidden_nodes}
    w2_shape = {w2_rows, n_output_classes}
    key1 = Nx.Random.key(57)
    {w1, key2} = Nx.Random.normal(key1, 0.0, 1, shape: w1_shape)
    {w2, _} = Nx.Random.normal(key2, 0.0, 1, shape: w2_shape)
    w1 = Nx.multiply(w1, Nx.sqrt(Nx.divide(1, w1_rows)))
    w2 = Nx.multiply(w2, Nx.sqrt(Nx.divide(1, w2_rows)))
    {w1, w2}
  end

  def train(x_train, y_train, x_test, y_test, n_hidden_nodes, iterations, lr) do
    n_input_variables = Nx.axis_size(x_train, 1)
    n_output_classes = Nx.axis_size(y_train, 1)
    {w1, w2} = initialize_weights(n_input_variables, n_hidden_nodes, n_output_classes)

    Enum.reduce(1..iterations, {w1, w2}, fn iter, {w1, w2} ->
      {y_hat, h} = forward(x_train, w1, w2)
      {w1_gradient, w2_gradient} = back(x_train, y_train, y_hat, w2, h)
      w1 = Nx.subtract(w1, Nx.multiply(w1_gradient, lr))
      w2 = Nx.subtract(w2, Nx.multiply(w2_gradient, lr))
      report(iter, x_train, y_train, x_test, y_test, w1, w2)
      {w1, w2}
    end)
  end
end
