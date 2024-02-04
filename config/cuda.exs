import Config

config :nx, :default_backend, EXLA.Backend
config :nx, :default_defn_options, compiler: EXLA

# https://hexdocs.pm/exla/EXLA.html#module-clients
config :exla, :default_client, :cuda

config :exla, :clients,
  cuda: [platform: :cuda],
  host: [platform: :host]
