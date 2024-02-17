defmodule ExLearning.MixProject do
  use Mix.Project

  def project do
    [
      app: :ex_learning,
      version: "0.1.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      preferred_cli_env: ["test.watch": :test]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {ExLearning.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.6.4"},
      {:exla, "~> 0.6.4"},
      {:axon, "~> 0.6.0"},
      {:mix_test_watch, "~> 1.1", only: [:dev, :test], runtime: false},
      {:table_rex, "~> 3.1.1"}
    ]
  end
end
