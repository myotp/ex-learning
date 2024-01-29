defmodule ExLearning.Core.LinearRegressionWithBias do
  import Nx.Defn

  def train(x, y, iterations, lr) do
    start_weight = 0
    start_bias = 0

    Enum.reduce_while(1..iterations, {start_weight, start_bias}, fn i, {w, b} ->
      current_loss = loss(x, y, w, b)
      IO.puts("Iteration #{i} => loss: #{current_loss}")

      cond do
        # 非GD方式, 只是简单演示的目的
        loss(x, y, w + lr, b) < current_loss ->
          {:cont, {w + lr, b}}

        loss(x, y, w - lr, b) < current_loss ->
          {:cont, {w - lr, b}}

        loss(x, y, w, b + lr) < current_loss ->
          {:cont, {w, b + lr}}

        loss(x, y, w, b - lr) < current_loss ->
          {:cont, {w, b - lr}}

        true ->
          {:halt, {w, b}}
      end
    end)
  end

  def loss(x, y, w, b) do
    nx_loss(x, y, w, b)
    |> Nx.to_number()
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
