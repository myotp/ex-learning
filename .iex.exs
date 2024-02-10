alias ExLearning.Projects.Pizza
alias ExLearning.Projects.MultiPizza
alias ExLearning.Projects.Police
alias ExLearning.Projects.Digit5
alias ExLearning.Projects.MnistRegression
alias ExLearning.Projects.MnistNN
alias ExLearning.Projects.MnistAxon
alias ExLearning.Projects.MnistCnn

defmodule ExLearning.IExHelpers do
  defdelegate r(), to: IEx.Helpers, as: :recompile
end

import ExLearning.IExHelpers
