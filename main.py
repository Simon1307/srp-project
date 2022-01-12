import tensorflow as tf
from data_utils import get_gan_dataset
from generator import Generator
from discriminator import Monet_Discriminator, Monet_Output_Layer, Photo_Discriminator
from model import CycleGan
from loss_functions import generator_loss1, generator_loss2, discriminator_loss1, discriminator_loss2
from loss_functions import cycle_loss, identity_loss
from diffaugment import aug_fn


def main():

    batch_size = 16

    monet_filepath = '/home/simon/Desktop/Uni/SRP/gan-getting-started/monet_tfrec/*.tfrec'
    photo_filepath = '/home/simon/Desktop/Uni/SRP/gan-getting-started/photo_tfrec/*.tfrec'
    monet_filenames = tf.io.gfile.glob(str(monet_filepath))
    photo_filenames = tf.io.gfile.glob(str(photo_filepath))

    final_dataset = get_gan_dataset(monet_filenames, photo_filenames, augment=None, repeat=True, shuffle=True,
                                    batch_size=batch_size)

    monet_generator = Generator()  # transforms photos to Monet-esque paintings
    photo_generator = Generator()  # transforms Monet paintings to be more like photos

    monet_discriminator = Monet_Discriminator()  # differentiates real Monet paintings and generated Monet paintings
    output1 = Monet_Output_Layer()  # learned with loss function awarding ...
    output2 = Monet_Output_Layer()  # learned with loss function awarding ---
    photo_discriminator = Photo_Discriminator()  # differentiates real photos and generated photos

    monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    cycle_gan_model = CycleGan(monet_generator=monet_generator,
                               photo_generator=photo_generator,
                               monet_discriminator=monet_discriminator,
                               photo_discriminator=photo_discriminator,
                               output_layer1=output1,
                               output_layer2=output2,
                               lambda_cycle=3,
                               lambda_id=3,
                               m_gen_optimizer=monet_generator_optimizer,
                               p_gen_optimizer=photo_generator_optimizer,
                               m_disc_optimizer=monet_discriminator_optimizer,
                               p_disc_optimizer=photo_discriminator_optimizer,
                               gen_loss_fn1=generator_loss1,
                               gen_loss_fn2=generator_loss2,
                               disc_loss_fn1=discriminator_loss1,
                               disc_loss_fn2=discriminator_loss2,
                               cycle_loss_fn=cycle_loss,
                               identity_loss_fn=identity_loss,
                               aug_fn=aug_fn
                               )
    cycle_gan_model.compile()
    # steps per epoch = ceil(num_samples / batch_size)
    cycle_gan_model.fit(final_dataset, steps_per_epoch=1, epochs=3, verbose=1)


if __name__ == '__main__':
    main()

