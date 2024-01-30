defmodule ExLearning.Projects.Police do
  alias ExLearning.Core.BinaryClassifier

  def run() do
    {input_params, police} = load_data()
    weight = BinaryClassifier.train(input_params, police, 10000, 0.001)
    IO.puts("最终训练得到m=#{Nx.squeeze(weight) |> Nx.to_list() |> inspect()}")
  end

  def load_data() do
    data = load_file() |> Enum.map(fn row -> List.insert_at(row, 0, 1) end)
    len = Enum.count(data)
    t = Nx.tensor(data)

    input_params = Nx.slice(t, [0, 0], [len, 4])

    police =
      t
      |> Nx.transpose()
      |> Nx.slice_along_axis(4, 1)
      |> Nx.reshape({len, 1})

    {input_params, police}
  end

  defp load_file() do
    __ENV__.file()
    |> Path.dirname()
    |> Path.join("data/police.txt")
    |> File.read!()
    |> String.split("\n", trim: true)
    |> Enum.drop(1)
    |> Enum.map(&String.split/1)
    |> Enum.map(fn nums -> Enum.map(nums, &String.to_integer/1) end)
  end
end
