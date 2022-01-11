import tensorflow as tf


class CycleGan(tf.keras.Model):
    def __init__(
            self,
            monet_generator,
            photo_generator,
            monet_discriminator,
            photo_discriminator,
            output_layer1,
            output_layer2,
            lambda_cycle=3,
            lambda_id=3,
    ):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id
        self.output_layer1 = output_layer1
        self.output_layer2 = output_layer2

    def compile(
            self,
            m_gen_optimizer,
            p_gen_optimizer,
            m_disc_optimizer,
            p_disc_optimizer,
            gen_loss_fn1,
            gen_loss_fn2,
            disc_loss_fn1,
            disc_loss_fn2,
            cycle_loss_fn,
            identity_loss_fn,
            aug_fn,

    ):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn1 = gen_loss_fn1
        self.gen_loss_fn2 = gen_loss_fn2
        self.disc_loss_fn1 = disc_loss_fn1
        self.disc_loss_fn2 = disc_loss_fn2
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        self.aug_fn = aug_fn
        self.step_num = 0

    def train_step(self, batch_data):
        print("inside train_step")
        real_monet, real_photo = batch_data
        batch_size = tf.shape(real_monet)[0]
        print("batch_size", batch_size)
        print("real_photo:", real_photo)
        with tf.GradientTape(persistent=True) as tape:
            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo, training=True)
            print("fake_monet", fake_monet)
            # only check whether fake monet corresponding to real_photo is too similar,
            # if id of real_photo is in list, meaning that last epoch disc was fooled by
            # fake monet generated by the real_photo
            # two-objective discriminator

            cycled_photo = self.p_gen(fake_monet, training=True)

            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # Diffaugment
            both_monet = tf.concat([real_monet, fake_monet], axis=0)

            aug_monet = self.aug_fn(both_monet)

            aug_real_monet = aug_monet[:batch_size]
            aug_fake_monet = aug_monet[batch_size:]

            disc_fake_monet1 = self.output_layer1(self.m_disc(aug_fake_monet, training=True), training=True)
            disc_real_monet1 = self.output_layer1(self.m_disc(aug_real_monet, training=True), training=True)
            disc_fake_monet2 = self.output_layer2(self.m_disc(aug_fake_monet, training=True), training=True)
            disc_real_monet2 = self.output_layer2(self.m_disc(aug_real_monet, training=True), training=True)

            monet_gen_loss1 = self.gen_loss_fn1(disc_fake_monet1)
            monet_output1_loss = self.disc_loss_fn1(disc_real_monet1, disc_fake_monet1)
            monet_gen_loss2 = self.gen_loss_fn2(disc_fake_monet2)
            monet_output2_loss = self.disc_loss_fn2(disc_real_monet2, disc_fake_monet2)

            monet_gen_loss = (monet_gen_loss1 + monet_gen_loss2) * 0.4
            monet_disc_loss = monet_output1_loss + monet_output2_loss

            # discriminator used to check, inputing real images

            disc_real_photo = self.p_disc(real_photo, training=True)
            # discriminator used to check, inputing fake images

            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # evaluates generator loss

            photo_gen_loss = self.gen_loss_fn1(disc_fake_photo)

            # evaluates discriminator loss

            photo_disc_loss = self.disc_loss_fn1(disc_real_photo, disc_fake_photo)

            # evaluates total generator loss
            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle / tf.cast(batch_size,
                                                                                                        tf.float32)) + self.cycle_loss_fn(
                real_photo, cycled_photo, self.lambda_cycle / tf.cast(batch_size, tf.float32))

            # evaluates total generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(real_monet, same_monet,
                                                                                             self.lambda_id / tf.cast(
                                                                                                 batch_size,
                                                                                                 tf.float32))
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo,
                                                                                             self.lambda_id / tf.cast(
                                                                                                 batch_size,
                                                                                                 tf.float32))

        # Calculate the gradients for generator and discriminator
        monet_generator_gradients = tape.gradient(total_monet_gen_loss,
                                                  self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        monet_discriminator_gradients = tape.gradient(monet_disc_loss,
                                                      self.m_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)

        # Heads gradients
        monet_head_gradients = tape.gradient(monet_output1_loss,
                                             self.output_layer1.trainable_variables)

        self.m_disc_optimizer.apply_gradients(zip(monet_head_gradients,
                                                  self.output_layer1.trainable_variables))

        monet_head_gradients = tape.gradient(monet_output2_loss,
                                             self.output_layer2.trainable_variables)
        self.m_disc_optimizer.apply_gradients(zip(monet_head_gradients,
                                                  self.output_layer2.trainable_variables))

        # Apply the gradients to the optimizer
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,
                                                 self.m_gen.trainable_variables))

        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,
                                                 self.p_gen.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients,
                                                  self.m_disc.trainable_variables))

        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,
                                                  self.p_disc.trainable_variables))

        return {
            "monet_output1_loss": monet_output1_loss,
            "monet_output2_loss2": monet_output2_loss,
            "disc_real_monet": disc_real_monet1,
            "disc_fake_monet": disc_fake_monet1,
            "disc_real_monet2": disc_real_monet2,
            "disc_fake_monet2": disc_fake_monet2,
            "monet_gen_loss": monet_gen_loss,
            "photo_disc_loss": photo_disc_loss,
        }
