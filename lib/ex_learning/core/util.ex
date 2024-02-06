defmodule ExLearning.Core.Util do
  def append_column(t, num) do
    rows = Nx.axis_size(t, 0)
    dummy_column = Stream.repeatedly(fn -> num end) |> Enum.take(rows)
    tt = Nx.transpose(t)
    t1 = Nx.tensor([dummy_column])

    Nx.concatenate([t1, tt])
    |> Nx.transpose()
  end
end
