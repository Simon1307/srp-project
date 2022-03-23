import tensorflow as tf


class CycleGan(tf.keras.Model):
    def __init__(
            self,
            monet_generator,
            photo_generator,
            monet_discriminator1,
            photo_discriminator,
            output_layer1,
            output_layer2,
            monet_discriminator2,

            output_layer3,
            output_layer4,
            lambda_cycle_photo,
            lambda_cycle_monet,
            lambda_id,
            lambda_guess,
            std,
            monet_discriminator_guess,
            photo_discriminator_guess,
            monet_discriminator_optimizer_guess,
            photo_discriminator_optimizer_guess,

            m_gen_optimizer,
            p_gen_optimizer,
            m_disc_optimizer,
            m_disc_optimizer2,
            p_disc_optimizer,

            gen_loss_fn1,
            gen_loss_fn2,
            disc_loss_fn1,
            disc_loss_fn2,
            cycle_loss_fn,
            identity_loss_fn,
            aug_fn,
    ):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc1 = monet_discriminator1
        self.p_disc = photo_discriminator
        self.output_layer1 = output_layer1
        self.output_layer2 = output_layer2
        self.m_disc2 = monet_discriminator2

        self.output_layer3 = output_layer3
        self.output_layer4 = output_layer4
        self.lambda_cycle_photo = lambda_cycle_photo
        self.lambda_cycle_monet = lambda_cycle_monet
        self.lambda_id = lambda_id
        self.lambda_guess = lambda_guess
        self.std = std
        self.m_disc_guess = monet_discriminator_guess
        self.p_disc_guess = photo_discriminator_guess
        self.monet_discriminator_optimizer_guess = monet_discriminator_optimizer_guess
        self.photo_discriminator_optimizer_guess = photo_discriminator_optimizer_guess

        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.m_disc_optimizer2 = m_disc_optimizer2
        self.p_disc_optimizer = p_disc_optimizer

        self.gen_loss_fn1 = gen_loss_fn1
        self.gen_loss_fn2 = gen_loss_fn2
        self.disc_loss_fn1 = disc_loss_fn1
        self.disc_loss_fn2 = disc_loss_fn2
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        self.aug_fn = aug_fn
        self.step_num = 0
        self.num_imgs_masked = tf.Variable(initial_value=0, dtype=tf.int32)
        self.num_imgs_fooled_disc = tf.Variable(initial_value=0, dtype=tf.int32)

    def compile(self):
        super(CycleGan, self).compile()

    def mod_p(self, mask, mask_p):
        self.num_imgs_masked.assign_add(1)
        return tf.where(mask, mask_p, 0)  # set corresponding index of fake Monet 0 at index in mask_p

    def mod_tracker(self, mask, monet):
        """Overwrite the tracker tensor storing the fake Monets that fooled the disc using
        tf.where.
        The mask tells which positions should be updated in the tracker tensor.
        The fake Monet is used to update the tensor"""

        self.num_imgs_fooled_disc.assign_add(1)
        mask = tf.reshape(mask, (7038, 1, 1, 1))
        mask = tf.repeat(mask, 256, axis=1)
        mask = tf.repeat(mask, 256, axis=2)
        mask = tf.repeat(mask, 3, axis=3)
        return tf.where(mask, self.tracker, monet)

    def return_same(self, x):
        return x

    def gaussian_noise(self, inp):
        """Adds noise to a given fake Monet"""
        noise = tf.random.normal(shape=tf.shape(inp),
                                 mean=0.0,
                                 stddev=self.std,
                                 dtype=tf.float32)
        return inp + noise

    def monet_guess_loss_false(self, real_monet, cycled_monet):
        concat_real_cycled_monet = tf.concat((real_monet, cycled_monet), 3)
        # correct_answ = False
        disc_monet_guess = self.m_disc_guess(concat_real_cycled_monet, training=True)
        # Guess discriminators are trained to minimize its error (fake, 0.0)
        disc_monet_guess_loss = self.gen_loss_fn2(disc_monet_guess) * self.lambda_guess
        # Generators are trained to produce images that maximize the guess loss (fake, 1.0)
        gen_monet_guess_loss = self.gen_loss_fn1(disc_monet_guess) * self.lambda_guess

        return disc_monet_guess_loss, gen_monet_guess_loss

    def monet_guess_loss_true(self, real_monet, cycled_monet):
        concat_real_cycled_monet = tf.concat((cycled_monet, real_monet), 3)
        # correct_answ = True
        disc_monet_guess = self.m_disc_guess(concat_real_cycled_monet, training=True)
        # (real, 1.0)
        disc_monet_guess_loss = self.gen_loss_fn1(disc_monet_guess) * self.lambda_guess
        # (real, 0.0)
        gen_monet_guess_loss = self.gen_loss_fn2(disc_monet_guess) * self.lambda_guess

        return disc_monet_guess_loss, gen_monet_guess_loss

    def photo_guess_loss_false(self, real_photo, cycled_photo):
        concat_real_cycled_photo = tf.concat((real_photo, cycled_photo), 3)
        # correct_answ = False
        disc_photo_guess = self.p_disc_guess(concat_real_cycled_photo, training=True)
        # (fake, 0.0)
        disc_photo_guess_loss = self.gen_loss_fn2(disc_photo_guess) * self.lambda_guess
        # (fake, 1.0)
        gen_photo_guess_loss = self.gen_loss_fn1(disc_photo_guess) * self.lambda_guess

        return disc_photo_guess_loss, gen_photo_guess_loss

    def photo_guess_loss_true(self, real_photo, cycled_photo):
        concat_real_cycled_photo = tf.concat((cycled_photo, real_photo), 3)
        # correct_answ = True
        disc_photo_guess = self.p_disc_guess(concat_real_cycled_photo, training=True)
        # (real, 1.0)
        disc_photo_guess_loss = self.gen_loss_fn1(disc_photo_guess) * self.lambda_guess
        # (real, 0.0)
        gen_photo_guess_loss = self.gen_loss_fn2(disc_photo_guess) * self.lambda_guess

        return disc_photo_guess_loss, gen_photo_guess_loss

    def train_step(self, batch_data):
        real_monet, photo_data = batch_data  # the data is split among every node of TPU
        real_photo, photo_ind = photo_data[0], photo_data[1]
        batch_size = tf.shape(real_monet)[0]

        batch_size_ = real_monet.shape[0]
        self.step_num += 1

        with tf.GradientTape(persistent=True) as tape:
            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo, training=True)
            # Add noise to fake monet
            fake_monet_noisy = self.gaussian_noise(fake_monet)
            cycled_photo = self.p_gen(fake_monet_noisy, training=True)

            # masking fake_monet
            mask_p = tf.ones((batch_size,))

            # tensor to store fake Monets which fooled disc
            self.tracker = tf.zeros((7038, 256, 256, 3))
            self.eyes = tf.eye(7038, dtype=tf.dtypes.bool)  # identity matrix
            eys_batch = tf.eye(batch_size, dtype=tf.dtypes.bool)  # identity matrix of batch size
            for i in range(batch_size_):
                # Calculate difference between current fake monet and respective stored old fake Monet
                # which fooled disc
                # the ssim score lays between 1 and -1 (1 indicates perfect structural similarity)
                ssim = tf.image.ssim(fake_monet[i], self.tracker[i], max_val=1.0, filter_size=11, filter_sigma=1.5,
                                     k1=0.01, k2=0.03)

                # If ssim score for fake Monet is above 0.5 (too similar to previous fake Monet) --> mask this fake monet
                mask_p = tf.cond(tf.math.greater(ssim, 0.5), lambda: self.mod_p(eys_batch[i], mask_p),
                                 lambda: self.return_same(mask_p))

            mask_p = tf.reshape(mask_p, (batch_size, 1, 1, 1))

            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            # Add noise to fake photo
            fake_photo_noisy = self.gaussian_noise(fake_photo)
            cycled_monet = self.m_gen(fake_photo_noisy, training=True)

            # generating itself
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # Diffaugment
            both_monet = tf.concat([real_monet, fake_monet], axis=0)
            aug_monet = self.aug_fn(both_monet)
            aug_real_monet = aug_monet[:batch_size]
            aug_fake_monet = aug_monet[batch_size:]

            # Discriminate augmented fake monets and augmented real monets using heads 1-4
            disc_fake_monet1 = self.output_layer1(self.m_disc1(aug_fake_monet, training=True), training=True)
            disc_real_monet1 = self.output_layer1(self.m_disc1(aug_real_monet, training=True), training=True)
            disc_fake_monet2 = self.output_layer2(self.m_disc1(aug_fake_monet, training=True), training=True)
            disc_real_monet2 = self.output_layer2(self.m_disc1(aug_real_monet, training=True), training=True)
            disc_fake_monet3 = self.output_layer3(self.m_disc2(aug_fake_monet, training=True), training=True)
            disc_real_monet3 = self.output_layer3(self.m_disc2(aug_real_monet, training=True), training=True)
            disc_fake_monet4 = self.output_layer4(self.m_disc2(aug_fake_monet, training=True), training=True)
            disc_real_monet4 = self.output_layer4(self.m_disc2(aug_real_monet, training=True), training=True)

            fool_thresh = tf.constant(0.3)
            output1_prob = tf.math.sigmoid(tf.math.reduce_mean(disc_fake_monet1, axis=(1, 2)))
            output2_prob = tf.math.sigmoid(tf.math.reduce_mean(disc_fake_monet2, axis=(1, 2)))
            output3_prob = tf.math.sigmoid(tf.math.reduce_mean(disc_fake_monet3, axis=(1, 2)))
            output4_prob = tf.math.sigmoid(tf.math.reduce_mean(disc_fake_monet4, axis=(1, 2)))

            output13_prob = tf.concat([output1_prob, output3_prob], axis=-1)
            output24_prob = tf.concat([output2_prob, output4_prob], axis=-1)
            output13_prob = tf.math.reduce_mean(output13_prob, axis=-1)
            output24_prob = tf.math.reduce_mean(output24_prob, axis=-1)

            for i in range(batch_size_):
                # Check for each fake Monet in batch whether it fooled the 4 heads
                self.tracker = tf.cond(
                    pred=tf.math.logical_and(output13_prob[i] > fool_thresh, output24_prob[i] < (1 - fool_thresh)),
                    true_fn=lambda: self.mod_tracker(self.eyes[photo_ind[i]], fake_monet[i]),
                    false_fn=lambda: self.return_same(self.tracker))

            # Discriminate real and fake photos using photo discriminator
            disc_real_photo = self.p_disc(real_photo, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # Calculate the loss of the monet generator, the monet discriminators, and their heads
            monet_gen_loss1 = self.gen_loss_fn1(disc_fake_monet1, mask_p)
            monet_output1_loss = self.disc_loss_fn1(disc_real_monet1, disc_fake_monet1, mask_p)
            monet_gen_loss2 = self.gen_loss_fn2(disc_fake_monet2, mask_p)
            monet_output2_loss = self.disc_loss_fn2(disc_real_monet2, disc_fake_monet2, mask_p)
            monet_gen_loss3 = self.gen_loss_fn1(disc_fake_monet3, mask_p)
            monet_output3_loss = self.disc_loss_fn1(disc_real_monet3, disc_fake_monet3, mask_p)
            monet_gen_loss4 = self.gen_loss_fn2(disc_fake_monet4, mask_p)
            monet_output4_loss = self.disc_loss_fn2(disc_real_monet4, disc_fake_monet4, mask_p)

            monet_gen_loss = (monet_gen_loss1 + monet_gen_loss2 + monet_gen_loss3 + monet_gen_loss4) * 0.4
            monet_disc_loss1 = monet_output1_loss + monet_output2_loss
            monet_disc_loss2 = monet_output3_loss + monet_output4_loss
            monet_disc_loss = monet_output1_loss + monet_output2_loss + monet_output3_loss + monet_output4_loss

            # Calculate loss of the photo generator and discriminator
            photo_gen_loss = self.gen_loss_fn1(disc_fake_photo)
            photo_disc_loss = self.disc_loss_fn1(disc_real_photo, disc_fake_photo)

            # Caclculate losses for discriminator m_disc_guess and m_gen using real_monet and cycled_monet
            # rand = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32, seed=123)
            # concat_threshold = tf.constant(0.5)
            # randomly decide the order of real and cycled monet
            # disc_monet_guess_loss, gen_monet_guess_loss = tf.cond(pred=(rand > concat_threshold),
            # true_fn=lambda: self.monet_guess_loss_false(real_monet, cycled_monet),
            # false_fn=lambda: self.monet_guess_loss_true(real_monet, cycled_monet))
            # randomly decide the order of real and cycled photo
            # disc_photo_guess_loss, gen_photo_guess_loss = tf.cond(pred=(rand > concat_threshold),
            # true_fn=lambda: self.photo_guess_loss_false(real_photo, cycled_photo),
            # false_fn=lambda: self.photo_guess_loss_true(real_photo, cycled_photo))

            # Total Cycle Loss
            monet_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet,
                                                  self.lambda_cycle_monet / tf.cast(batch_size, tf.float32))
            photo_cycle_loss = self.cycle_loss_fn(real_photo, cycled_photo,
                                                  self.lambda_cycle_photo / tf.cast(batch_size, tf.float32))
            total_cycle_loss = monet_cycle_loss + photo_cycle_loss

            # Identity loss
            monet_identity_loss = self.identity_loss_fn(real_monet, same_monet,
                                                        self.lambda_id / tf.cast(batch_size, tf.float32))
            photo_identity_loss = self.identity_loss_fn(real_photo, same_photo,
                                                        self.lambda_id / tf.cast(batch_size, tf.float32))

            # Calculate total monet and photo generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + monet_identity_loss  # + gen_monet_guess_loss + gen_photo_guess_loss
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + photo_identity_loss  # + gen_photo_guess_loss + gen_monet_guess_loss

        ######################################################
        # Calculate the gradients for generators, discriminators, and heads
        # Photo and Monet Generator
        monet_generator_gradients = tape.gradient(total_monet_gen_loss,
                                                  self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        # Monet Discriminators and Photo Discriminator
        monet_discriminator_gradients1 = tape.gradient(monet_disc_loss1,
                                                       self.m_disc1.trainable_variables)
        monet_discriminator_gradients2 = tape.gradient(monet_disc_loss2,
                                                       self.m_disc2.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)
        # print("photo disc gradients", photo_discriminator_gradients)
        # Guess Discriminators
        # monet_discriminator_guess_gradients = tape.gradient(disc_monet_guess_loss,
        #                                              self.m_disc_guess.trainable_variables)
        # photo_discriminator_guess_gradients = tape.gradient(disc_photo_guess_loss,
        #                                              self.p_disc_guess.trainable_variables)
        # print("Monet disc guess gradients", monet_discriminator_guess_gradients)
        # print("Photo disc guess gradients", photo_discriminator_guess_gradients)

        # Heads gradients
        monet_head_gradients1 = tape.gradient(monet_output1_loss,
                                              self.output_layer1.trainable_variables)
        monet_head_gradients2 = tape.gradient(monet_output2_loss,
                                              self.output_layer2.trainable_variables)
        monet_head_gradients3 = tape.gradient(monet_output3_loss,
                                              self.output_layer3.trainable_variables)
        monet_head_gradients4 = tape.gradient(monet_output4_loss,
                                              self.output_layer4.trainable_variables)

        ##############################################
        # Apply the gradients to the optimizer of generators, discriminators, and discriminator heads
        # Generators
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,
                                                 self.m_gen.trainable_variables))
        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,
                                                 self.p_gen.trainable_variables))
        # Discriminators
        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients1,
                                                  self.m_disc1.trainable_variables))
        self.m_disc_optimizer2.apply_gradients(zip(monet_discriminator_gradients2,
                                                   self.m_disc2.trainable_variables))
        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,
                                                  self.p_disc.trainable_variables))
        # Heads
        self.m_disc_optimizer.apply_gradients(zip(monet_head_gradients1,
                                                  self.output_layer1.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_head_gradients2,
                                                  self.output_layer2.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_head_gradients3,
                                                  self.output_layer3.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_head_gradients4,
                                                  self.output_layer4.trainable_variables))
        # Guess Discriminators
        # self.monet_discriminator_optimizer_guess.apply_gradients(zip(monet_discriminator_guess_gradients,
        #                                            self.m_disc_guess.trainable_variables))
        # self.photo_discriminator_optimizer_guess.apply_gradients(zip(photo_discriminator_guess_gradients,
        #                                            self.p_disc_guess.trainable_variables))

        return {
            "monet_gen_loss1": monet_gen_loss1,
            "monet_gen_loss2": monet_gen_loss2,
            "monet_gen_loss3": monet_gen_loss3,
            "monet_gen_loss4": monet_gen_loss4,
            "monet_output1_loss": monet_output1_loss,
            "monet_output2_loss": monet_output2_loss,
            "monet_output3_loss": monet_output3_loss,
            "monet_output4_loss": monet_output4_loss,
            "monet_disc_loss1": monet_disc_loss1,
            "monet_disc_loss2": monet_disc_loss2,
            "monet_gen_loss": monet_gen_loss,
            "photo_gen_loss": photo_gen_loss,
            "photo_disc_loss": photo_disc_loss,
            "Total cycle loss": total_cycle_loss,
            # "photo_disc_loss_guess": disc_photo_guess_loss,
            # "monet_disc_loss_guess": disc_monet_guess_loss,
            # "gen_monet_guess_loss": gen_monet_guess_loss,
            # "gen_photo_guess_loss": gen_photo_guess_loss,
            "output1_prob": output1_prob,
            "output2_prob": output2_prob,
            "output3_prob": output3_prob,
            "output4_prob": output4_prob,
            "output13_prob": output13_prob,
            "output24_prob": output24_prob,
            "num_imgs_masked": self.num_imgs_masked,
            "num_imgs_fooled_disc": self.num_imgs_fooled_disc
        }
