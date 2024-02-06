defmodule ExLearning.Core.Util do
  import Nx.Defn

  defn prepend_column(t, num) do
    rows = Nx.axis_size(t, 0)
    t0 = Nx.broadcast(num, {1, rows})

    Nx.concatenate([t0, Nx.transpose(t)])
    |> Nx.transpose()
  end
end
