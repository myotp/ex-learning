defmodule ExLearning.Core.BinaryClassifier do
  import Nx.Defn

  defn sigmoid(z) do
    1 / (1 + Nx.exp(-z))
  end

  defn forward(x, w) do
    Nx.dot(x, w)
    |> sigmoid()
  end

  defn classify(x, w) do
    forward(x, w)
    |> Nx.round()
  end

  # logistic loss
  defn log_loss(x, y, w) do
    y_hat = forward(x, w)
    first_term = y * Nx.log(y_hat)
    second_term = (1 - y) * Nx.log(1 - y_hat)
    -Nx.mean(first_term + second_term)
  end

  def loss(x, y, w) do
    log_loss(x, y, w)
    |> Nx.to_number()
  end

  defn gradient(x, y, w) do
    x_t = Nx.transpose(x)
    x_shape_0 = Nx.axis_size(x, 0)
    Nx.dot(x_t, forward(x, w) - y) / x_shape_0
  end

  # 这里train函数与gradient_descent与前边multi_regression一模一样
  # 所以, 实际当中只要传入一个回调的loss/3与gradient/3函数就行了
  def train(x, y, iterations, lr) do
    x_shape_1 = Nx.axis_size(x, 1)
    start_weight = Nx.iota({x_shape_1, 1}, axis: 1)

    Enum.reduce(1..iterations, start_weight, fn i, weight ->
      current_loss = loss(x, y, weight)
      IO.puts("Iteration #{i} => loss: #{current_loss}")
      g = gradient(x, y, weight)
      gradient_descent(weight, g, lr)
    end)
  end

  defn gradient_descent(weight, gradient, learning_rate) do
    weight - gradient * learning_rate
  end
end
