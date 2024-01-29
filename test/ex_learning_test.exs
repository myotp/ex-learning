defmodule ExLearningTest do
  use ExUnit.Case
  doctest ExLearning

  test "greets the world" do
    assert ExLearning.hello() == :world
  end
end
