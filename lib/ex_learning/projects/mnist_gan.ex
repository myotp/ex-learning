defmodule ExLearning.Projects.MnistGAN do
  alias ExLearning.Axon.MnistGAN

  @train_image_file "train-images-idx3-ubyte.gz"

  def run() do
    train_images = load_train_images()
    generator = MnistGAN.build_generator(100)
    discriminator = MnistGAN.build_discriminator({nil, 28, 28, 1})
    MnistGAN.run(discriminator, generator, train_images)
  end

  def load_train_images() do
    <<_::32, n_images::32, n_rows::32, n_cols::32, images_bin::binary>> =
      read_and_unzip!(@train_image_file)

    # 在CUDA机器上面跑这种操作很快
    images_bin
    |> Nx.from_binary({:u, 8})
    |> Nx.reshape({n_images, n_rows, n_cols, 1})
    |> Nx.divide(255)
    |> Nx.to_batched(32)
  end

  def read_and_unzip!(filename) do
    filename
    |> File.read!()
    |> :zlib.gunzip()
  end
end
