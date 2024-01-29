alias ExLearning.Projects.Pizza

defmodule ExLearning.IExHelpers do
  defdelegate r(), to: IEx.Helpers, as: :recompile
end

import ExLearning.IExHelpers
