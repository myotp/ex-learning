defmodule ExLearning.Core.LinearRegression do
  import Nx.Defn

  def train(x, y, iterations, learning_rate) do
    start_weight = 0

    Enum.reduce_while(1..iterations, start_weight, fn i, weight ->
      current_loss = loss(x, y, weight)
      IO.puts("Iteration #{i} => loss: #{current_loss}")

      cond do
        # 非GD方式, 只是简单演示的目的
        loss(x, y, weight + learning_rate) < current_loss ->
          {:cont, weight + learning_rate}

        loss(x, y, weight - learning_rate) < current_loss ->
          {:cont, weight - learning_rate}

        true ->
          {:halt, weight}
      end
    end)
  end

  def loss(x, y, w) do
    nx_loss(x, y, w)
    |> Nx.to_number()
  end

  # 最简单的不含bias的线性函数y=weight*x
  defn predict(x, weight) do
    weight * x
  end

  # Mean Squared Errors (MSE)
  defn nx_loss(x, y, w) do
    y_pred = predict(x, w)

    Nx.pow(y_pred - y, 2)
    |> Nx.mean()
  end
end
