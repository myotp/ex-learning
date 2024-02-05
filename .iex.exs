alias ExLearning.Projects.Pizza
alias ExLearning.Projects.MultiPizza
alias ExLearning.Projects.Police
alias ExLearning.Projects.Digit5
alias ExLearning.Projects.Mnist

defmodule ExLearning.IExHelpers do
  defdelegate r(), to: IEx.Helpers, as: :recompile
end

import ExLearning.IExHelpers
