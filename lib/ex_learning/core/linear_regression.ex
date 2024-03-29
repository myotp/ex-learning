defmodule ExLearning.Core.LinearRegression do
  import Nx.Defn

  def train(x, y, iterations, lr) do
    start_weight = 0
    start_bias = 0

    Enum.reduce(1..iterations, {start_weight, start_bias}, fn i, {w, b} ->
      current_loss = loss(x, y, w, b)
      IO.puts("Iteration #{i} => loss: #{current_loss}")
      {gw, gb} = gradient(x, y, w, b)
      {gw, gb} = {Nx.to_number(gw), Nx.to_number(gb)}
      weight = w - gw * lr
      bias = b - gb * lr
      {weight, bias}
    end)
  end

  def loss(x, y, w, b) do
    nx_loss(x, y, w, b)
    |> Nx.to_number()
  end

  # 针对MSE的loss函数gradient函数
  defn gradient(x, y, w, b) do
    grad_w = 2 * Nx.mean(x * (predict(x, w, b) - y))
    grad_b = 2 * Nx.mean(predict(x, w, b) - y)
    {grad_w, grad_b}
  end

  # 最简单的不含bias的线性函数y=weight*x
  defn predict(x, weight, bias) do
    weight * x + bias
  end

  # Mean Squared Errors (MSE)
  defn nx_loss(x, y, w, b) do
    y_pred = predict(x, w, b)

    Nx.pow(y_pred - y, 2)
    |> Nx.mean()
  end
end
