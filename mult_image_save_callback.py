
import os
from keras.callbacks import Callback
from matplotlib import pyplot as plt
import numpy as np
import scipy


class ImageSaveCallback(Callback):
    '''
    Image callback to save example synthetic images, latent representation images, training stats and a Keras model
    for every epoch.
    '''
    def __init__(self, train_inputs, train_outputs, val_inputs, val_outputs, folder_name, output_modalities):
        super(ImageSaveCallback, self).__init__()

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        self.folder_name = folder_name

        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.val_inputs = val_inputs
        self.val_outputs = val_outputs

        self.output_modalities = output_modalities

        self.train_losses = dict()
        self.val_losses = dict()

        num_emb = len(self.train_inputs) + 1
        self.losses = ['val_loss'] + ['val_em_%d_dec_%s_loss' % (em, mod) for em in range(num_emb)
                                      for mod in self.output_modalities] + ['val_em_concat_loss']
        csv_header = 'epochs, ' + ', '.join(self.losses) + '\n'
        self.csv_file = self.folder_name + '/training_results.csv'
        fd = open(self.csv_file, "w")
        fd.write(csv_header)
        fd.close()

    def saveImage(self, vol, slice_ids, filename, inputs, outputs):
        # saves a results image for the given source and target volumes at the given slices
        x_vol = [tr_x[vol] for tr_x in inputs if vol < len(tr_x)]
        y_vol = [tr_y[vol] for tr_y in outputs if vol < len(tr_y)]

        if len(x_vol) == 0 or len(y_vol) == 0:
            return

        z_vol = self.model.predict(x_vol)

        if x_vol[0].shape[1] > 1:
            print 'Reformatting callback data'
            sh = x_vol[0].shape
            x_vol = [xv[:, sh[1] / 2:sh[1] / 2 + 1] for xv in x_vol]
            assert x_vol[0].shape[1] == 1

        if y_vol[0].shape[1] > 1:
            sh = y_vol[0].shape
            y_vol = [yv[:, sh[1] / 2:sh[1] / 2 + 1] for yv in y_vol]
            assert y_vol[0].shape[1] == 1

        num_inputs = len(x_vol)
        im_shape = z_vol[0][0, 0].shape

        for sl_id in slice_ids:
            rows = [
                np.concatenate([x_vol[xi][sl_id, 0] for xi in range(num_inputs)] + 2 * [np.zeros(im_shape)], axis=1)]
            for yi, y in enumerate(y_vol):
                # outputs in the form of em_0_dec_VFlair, em_1_dec_VFlair, ..., em_0_dec_T1, ...
                z_idx = [outi for outi, out in enumerate(self.model.outputs) if self.output_modalities[yi] in out.name]
                y_synth = [z_vol[zi][sl_id, 0] for zi in z_idx]
                if len(y_synth) == 1:
                    y_synth = num_inputs * [np.zeros(im_shape)] + y_synth
                elif len(y_synth) == num_inputs:
                    y_synth += [np.zeros(im_shape)]
                y_truth = y[sl_id, 0]
                rows.append(np.concatenate(y_synth + [y_truth], axis=1))
                error = [np.abs(ys - y_truth) for ys in y_synth]
                rows.append(np.concatenate(error + [np.zeros(im_shape)], axis=1))

            img_array = np.concatenate(rows, axis=0)
            scipy.misc.imsave(filename + '_' + str(sl_id) + '.png', img_array)

        if num_inputs > 1:
            try:
                num_emb = z_vol[-2].shape[1] + 1
                latent_dim = z_vol[-2].shape[2] / (im_shape[0] * im_shape[1])
                emb_flatten = [np.expand_dims(emb, axis=2) for emb in [z_vol[-2], z_vol[-1]]]
                emb_flatten = np.concatenate(emb_flatten, axis=1)
                emb_shape = emb_flatten.shape[0], emb_flatten.shape[1], latent_dim * im_shape[0], im_shape[1]
                embeddings = np.reshape(emb_flatten, emb_shape)
                embeddings_ims = np.concatenate([embeddings[96, j] for j in range(num_emb)], axis=1)
                scipy.misc.imsave(filename + '_embeddings.png', embeddings_ims)
            except:
                print 'Skipping embedding visualisation'

    def save_examples(self, img_size='small', name_prefix=''):
        # saves examples for some layers of the 0th training and 0th validation volumes

        if img_size == 'small':
            slice_ids = [72, 108]
        elif img_size == 'big':
            slice_ids = [60, 72, 96, 108]
        else:
            raise Exception('Illegal value for img_size ', img_size)

        fn = self.folder_name + '/%sres_training' % name_prefix
        self.saveImage(0, slice_ids, fn + '_0', self.train_inputs, self.train_outputs)
        self.saveImage(17, [72, 96], fn + '_17', self.train_inputs, self.train_outputs)

        fn = self.folder_name + '/%sres_validation' % name_prefix
        self.saveImage(0, slice_ids, fn + '_0', self.val_inputs, self.val_outputs)
        self.saveImage(1, slice_ids, fn + '_1', self.val_inputs, self.val_outputs)

    def on_epoch_end(self, epoch, logs=None):

        # save the model:
        if logs is None:
            logs = {}
        self.model.save(self.folder_name + '/model')
        self.model.save(self.folder_name + '/model_%d' % epoch)
        for i in range(0, epoch - 2):
            if os.path.exists(self.folder_name + '/model_%d' % i):
                os.remove(self.folder_name + '/model_%d' % i)

        # save some larger example images at the end of each epoch:
        self.save_examples(name_prefix='epoch_end_', img_size='big')

        self.save_loss(epoch, logs)

    def save_loss(self, epoch, logs):
        fd = open(self.csv_file, "a")
        row = str(epoch) + ',' + ','.join([str(logs[l]) for l in self.losses if l in logs]) + '\n'
        fd.write(row)
        fd.close()

        plt.figure()
        for l in logs:
            if 'val' in l:
                continue

            if l not in self.train_losses:
                self.train_losses[l] = []
            self.train_losses[l].append(logs[l])
            plt.plot(self.train_losses[l], label=l)
        plt.legend()
        plt.title('Training Loss')
        plt.savefig(self.folder_name + '/resplot_train.png')
        plt.clf()

        plt.figure()
        for l in logs:
            if 'val' not in l:
                continue

            if l not in self.val_losses:
                self.val_losses[l] = []
            self.val_losses[l].append(logs[l])
            plt.plot(self.val_losses[l], label=l)
        plt.title('Validation Loss')
        plt.legend()
        plt.savefig(self.folder_name + '/resplot_validation.png')
        plt.clf()
