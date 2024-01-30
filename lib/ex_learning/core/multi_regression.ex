defmodule ExLearning.Core.MultiRegression do
  import Nx.Defn

  # 替换普通乘法*为Nx.dot才是做两个矩阵(tensor)相乘
  defn predict(x, weight) do
    Nx.dot(x, weight)
  end

  # loss函数同样直接linear regression版本自动工作了已经
  defn nx_loss(x, y, w) do
    y_pred = predict(x, w)

    Nx.pow(y_pred - y, 2)
    |> Nx.mean()
  end

  def train(x, y, iterations, learning_rate) do
    x_shape_1 = Nx.axis_size(x, 1)
    start_weight = Nx.iota({x_shape_1, 1}, axis: 1)
    IO.inspect(start_weight, label: "起始随机weight矩阵")
    # 用gradient descent方式快速求解, 不再需要分别尝试增大减小w了
    # 导数趋近于0但是永远不等于0最终
    Enum.reduce(1..iterations, start_weight, fn i, weight ->
      current_loss = loss(x, y, weight)
      IO.puts("Iteration #{i} => loss: #{current_loss}")
      g = gradient(x, y, weight)
      gradient_descent(weight, g, learning_rate)
    end)
  end

  defn gradient_descent(weight, gradient, learning_rate) do
    weight - gradient * learning_rate
  end

  def loss(x, y, w) do
    nx_loss(x, y, w)
    |> Nx.to_number()
  end

  # Multi linear regression的gradient有点复杂
  defn gradient(x, y, w) do
    2 * Nx.dot(Nx.transpose(x), predict(x, w) - y) / Nx.axis_size(x, 0)
  end
end
