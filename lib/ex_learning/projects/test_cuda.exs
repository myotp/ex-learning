# https://elixirforum.com/t/can-anyone-post-a-link-to-a-livebook-that-uses-nvidia-cuda-to-run-simple-nx-benchmarks/60403
# https://github.com/shanesveller/advent-of-code/blob/elixir-2023/cuda-test.livemd
Mix.install(
  [
    {:nx, "~> 0.6.4"},
    {:exla, "~> 0.6.4"}
  ],
  config: [
    nx: [
      default_backend: EXLA.Backend,
      default_defn_options: [compiler: EXLA]
    ],
    exla: [
      default_client: :cuda,
      clients: [
        host: [platform: :host],
        cuda: [platform: :cuda]
      ]
    ]
  ],
  system_env: [
    XLA_TARGET: "cuda120"
  ]
)

Nx.with_default_backend({EXLA.Backend, client: :cuda}, fn ->
  Nx.iota({10, 10})
  |> Nx.add(10)
end)
