defmodule ExLearning.Projects.Pizza do
  alias ExLearning.Core.LinearRegression
  alias ExLearning.Core.LinearRegressionWithBias

  def run() do
    {x, y} = load_data()
    weight = LinearRegression.train(x, y, 10000, 0.001)
    IO.puts("最终训练得到m=#{weight}")
  end

  def run_with_bias() do
    {x, y} = load_data()
    {weight, bias} = LinearRegressionWithBias.train(x, y, 10000, 0.001)
    IO.puts("最终训练得到m=#{weight} b=#{bias}")
  end

  def load_data() do
    t =
      load_file()
      |> Nx.tensor()
      |> Nx.transpose()

    reservations = Nx.slice_along_axis(t, 0, 1)
    sold_pizzas = Nx.slice_along_axis(t, 1, 1)
    {reservations, sold_pizzas}
  end

  defp load_file() do
    __ENV__.file()
    |> Path.dirname()
    |> Path.join("data/pizza.txt")
    |> File.read!()
    |> String.split("\r\n", trim: true)
    |> Enum.drop(1)
    |> Enum.map(&String.split/1)
    |> Enum.map(fn nums -> Enum.map(nums, &String.to_integer/1) end)
  end
end
