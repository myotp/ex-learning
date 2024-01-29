defmodule ExLearning.Core.LinearRegression do
  import Nx.Defn

  def train(x, y, iterations, learning_rate) do
    start_weight = 0

    # 用gradient descent方式快速求解, 不再需要分别尝试增大减小w了
    # 导数趋近于0但是永远不等于0最终
    Enum.reduce(1..iterations, start_weight, fn i, weight ->
      current_loss = loss(x, y, weight)
      IO.puts("Iteration #{i} => loss: #{current_loss}")
      g = Nx.to_number(gradient(x, y, weight))
      weight - g * learning_rate
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

  # 针对MSE的loss函数gradient函数
  defn gradient(x, y, w) do
    2 * Nx.mean(x * (predict(x, w) - y))
  end

  # Mean Squared Errors (MSE)
  defn nx_loss(x, y, w) do
    y_pred = predict(x, w)

    Nx.pow(y_pred - y, 2)
    |> Nx.mean()
  end
end
