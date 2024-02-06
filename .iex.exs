alias ExLearning.Projects.Pizza
alias ExLearning.Projects.MultiPizza
alias ExLearning.Projects.Police
alias ExLearning.Projects.Digit5
alias ExLearning.Projects.Mnist
alias ExLearning.Projects.MnistNN

defmodule ExLearning.IExHelpers do
  defdelegate r(), to: IEx.Helpers, as: :recompile
end

import ExLearning.IExHelpers
