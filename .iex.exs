alias ExLearning.Projects.Pizza
alias ExLearning.Projects.MultiPizza

defmodule ExLearning.IExHelpers do
  defdelegate r(), to: IEx.Helpers, as: :recompile
end

import ExLearning.IExHelpers
