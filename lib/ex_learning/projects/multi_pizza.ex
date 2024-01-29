defmodule ExLearning.Projects.MultiPizza do
  alias ExLearning.Core.MultiRegressionWithoutBias

  def run_without_bias() do
    {x, y} = load_data()
    weight = MultiRegressionWithoutBias.train(x, y, 5_000, 0.001)
    IO.puts("最终训练得到m=#{Nx.squeeze(weight) |> Nx.to_list() |> inspect()}")
  end

  def load_data() do
    t =
      load_file()
      |> Nx.tensor()

    input_params = Nx.slice(t, [0, 0], [30, 3])

    sold_pizzas =
      t
      |> Nx.transpose()
      |> Nx.slice_along_axis(3, 1)
      |> Nx.reshape({30, 1})

    {input_params, sold_pizzas}
  end

  defp load_file() do
    __ENV__.file()
    |> Path.dirname()
    |> Path.join("data/pizza_3_vars.txt")
    |> File.read!()
    |> String.split("\n", trim: true)
    |> Enum.drop(1)
    |> Enum.map(&String.split/1)
    |> Enum.map(fn nums -> Enum.map(nums, &String.to_integer/1) end)
  end
end
