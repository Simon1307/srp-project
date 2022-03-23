import tensorflow as tf
from data_utils import get_gan_dataset
from generator import Generator
from discriminator import Monet_Discriminator, Monet_Output_Layer, Photo_Discriminator, Discriminator_Guess
from model import CycleGan
from loss_functions import generator_loss1, generator_loss2, discriminator_loss1, discriminator_loss2
from loss_functions import cycle_loss, identity_loss
from diffaugment import aug_fn


def main():

    batch_size = 16  # 64
    epochs = 2  # 250
    lambda_cycle_photo = 2.6
    lambda_cycle_monet = 3
    lambda_id = 3
    lambda_guess = 0.5
    std = 0.05
    initial_learning_rate = 0.0002
    decay_steps = 12000
    decay_rate = 0.75
    beta_1 = 0.5
    steps_per_epoch = 109

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)

    monet_filepath = '/home/simon/Desktop/Uni/SRP/gan-getting-started/monet_tfrec/*.tfrec'
    photo_filepath = '/home/simon/Desktop/Uni/SRP/gan-getting-started/photo_tfrec/*.tfrec'
    monet_filenames = tf.io.gfile.glob(str(monet_filepath))
    photo_filenames = tf.io.gfile.glob(str(photo_filepath))

    final_dataset = get_gan_dataset(monet_filenames, photo_filenames, augment=None, repeat=True, shuffle=True,
                                    batch_size=batch_size)

    monet_generator = Generator()  # transforms photos to Monet-esque paintings
    photo_generator = Generator()  # transforms Monet paintings to be more like photos

    monet_discriminator1 = Monet_Discriminator()  # differentiates real Monet paintings and generated Monet paintings
    output1 = Monet_Output_Layer()  # learned with loss function awarding high scores for real and low scores for fake images
    output2 = Monet_Output_Layer()  # learned with loss function awarding low scores for real and high scores for fake images
    monet_discriminator2 = Monet_Discriminator()  # differentiates real Monet paintings and generated Monet paintings
    output3 = Monet_Output_Layer()  # learned with loss function awarding high scores for real and low scores for fake images
    output4 = Monet_Output_Layer()  # learned with loss function awarding low scores for real and high scores for fake images
    photo_discriminator = Photo_Discriminator()  # differentiates real photos and generated photos

    monet_discriminator_guess = Discriminator_Guess()
    photo_discriminator_guess = Discriminator_Guess()

    monet_generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=beta_1)
    photo_generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=beta_1)

    monet_discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=beta_1)
    monet_discriminator_optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=beta_1)
    photo_discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=beta_1)

    monet_discriminator_optimizer_guess = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=beta_1)
    photo_discriminator_optimizer_guess = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=beta_1)

    cycle_gan_model = CycleGan(monet_generator=monet_generator,
                               photo_generator=photo_generator,
                               monet_discriminator1=monet_discriminator1,
                               monet_discriminator2=monet_discriminator2,
                               photo_discriminator=photo_discriminator,
                               output_layer1=output1,
                               output_layer2=output2,
                               output_layer3=output3,
                               output_layer4=output4,
                               lambda_cycle_photo=lambda_cycle_photo,
                               lambda_cycle_monet=lambda_cycle_monet,
                               lambda_id=lambda_id,
                               lambda_guess=lambda_guess,
                               std=std,
                               monet_discriminator_guess=monet_discriminator_guess,
                               photo_discriminator_guess=photo_discriminator_guess,
                               monet_discriminator_optimizer_guess=monet_discriminator_optimizer_guess,
                               photo_discriminator_optimizer_guess=photo_discriminator_optimizer_guess,
                               m_gen_optimizer=monet_generator_optimizer,
                               p_gen_optimizer=photo_generator_optimizer,
                               m_disc_optimizer=monet_discriminator_optimizer,
                               m_disc_optimizer2=monet_discriminator_optimizer2,
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
    cycle_gan_model.fit(final_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs)


if __name__ == '__main__':
    main()

