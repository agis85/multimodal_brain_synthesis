import os
import traceback
import scipy
import numpy as np
import json

from keras.callbacks import EarlyStopping
from model import Multimodel
from mult_image_save_callback import ImageSaveCallback
from error_metrics import ErrorMetrics


class Experiment(object):
    def __init__(self, input_modalities, output_weights, folder_name, data, latent_dim=4, spatial_transformer=True,
                 common_merge='max', ind_outs=True, fuse_outs=True):
        self.input_modalities = input_modalities
        self.output_weights = output_weights
        self.output_modalities = sorted([o for o in output_weights if o != 'concat'])
        self.latent_dim = latent_dim
        self.folder_name = folder_name
        self.data = data
        self.spatial_transformer = spatial_transformer
        self.common_merge = common_merge
        self.ind_outs = ind_outs
        self.fuse_outs = fuse_outs
        assert ind_outs or fuse_outs
        self.mm = None

    def create_model(self):
        print('Creating model...')
        mod = self.input_modalities[0]
        chn = self.data.select_for_ids(mod, [0]).shape[1]
        print 'Channels: %d' % chn
        self.mm = Multimodel(self.input_modalities, self.output_modalities, self.output_weights, self.latent_dim, chn,
                             self.spatial_transformer, self.common_merge, self.ind_outs, self.fuse_outs)
        self.mm.build()

    def load_model(self, folder, model_name='model'):
        self.create_model()
        self.mm.model.load_weights(folder + '/' + model_name)

    def save(self, folder_name):
        print 'Saving experiment details'
        exp_json = {'input_modalities': self.input_modalities,
                    'output_weights': self.output_weights,
                    'latent_dim': self.latent_dim,
                    'model_layers': [l.name for l in self.mm.model.layers],
                    'encoder_params': [l.count_params() for l in self.mm.model.layers if
                                       'enc_' + self.input_modalities[0] in l.name]
                    }
        with open(folder_name + '/experiment_config.json', 'w') as f:
            json.dump(exp_json, f)

    # Run experiment for cross-validation
    def run(self, data):
        self.data = data

        for splid, split_dict in enumerate(data.id_splits_iterator()):
            print('Running for split ' + str(splid))

            folder_split = self.folder_name + '/split' + str(splid)
            if not os.path.exists(folder_split):
                try:
                    self.run_at_split(split_dict, folder_split)
                except Exception:
                    traceback.print_exc()

                try:
                    self.test_at_split(split_dict, folder_split)
                except Exception:
                    traceback.print_exc()

                self.save(folder_split)

    def run_at_split(self, split_dict, folder_split, model=None):
        ids_train = split_dict['train']
        ids_valid = split_dict['validation']

        if model is None:
            print('Creating model...')
            self.create_model()
        assert self.mm.model is not None

        initial_weights = [lay.get_weights() for lay in self.mm.model.layers]

        cb_train_in = [self.data.select_for_ids(mod, ids_train, as_array=False) for mod in self.input_modalities]
        cb_train_out = [self.data.select_for_ids(mod, ids_train, as_array=False) for mod in self.output_modalities]
        cb_valid_in = [self.data.select_for_ids(mod, ids_valid, as_array=False) for mod in self.input_modalities]
        cb_valid_out = [self.data.select_for_ids(mod, ids_valid, as_array=False) for mod in self.output_modalities]

        cb = ImageSaveCallback(cb_train_in, cb_train_out, cb_valid_in, cb_valid_out, folder_split,
                               self.output_modalities)

        es = EarlyStopping(monitor='val_loss', min_delta=0.01, mode='min', patience=10)

        train_in = [self.data.select_for_ids(mod, ids_train) for mod in self.input_modalities]
        valid_in = [self.data.select_for_ids(mod, ids_valid) for mod in self.input_modalities]

        # there's 1 output per embedding plus 1 output for the total variance embedding
        train_out = [self.data.select_for_ids(mod, ids_train) for mod in self.output_modalities
                     for i in range(self.mm.num_emb)]
        valid_out = [self.data.select_for_ids(mod, ids_valid) for mod in self.output_modalities
                     for i in range(self.mm.num_emb)]
        train_shape = (train_out[0].shape[0], 1, train_out[0].shape[2], train_out[0].shape[3])
        valid_shape = (valid_out[0].shape[0], 1, valid_out[0].shape[2], valid_out[0].shape[3])

        if train_out[0].shape[1] > 1:
            print 'Reformatting output data'
            sh = train_out[0].shape
            train_out = [to[:, sh[1] / 2:sh[1] / 2 + 1] for to in train_out]
            valid_out = [vo[:, sh[1] / 2:sh[1] / 2 + 1] for vo in valid_out]
            assert train_out[0].shape[1] == 1
            assert valid_out[0].shape[1] == 1

        if len(self.input_modalities) > 1:
            train_out += [np.zeros(shape=train_shape) for i in range(2)]
            valid_out += [np.zeros(shape=valid_shape) for i in range(2)]

        print 'Loss Weights'
        print self.mm.model.loss_weights

        print('Fitting model...')
        self.mm.model.fit(train_in, train_out, validation_data=(valid_in, valid_out), epochs=100, batch_size=16,
                          callbacks=[cb, es])

        final_weights = [lay.get_weights() for lay in self.mm.model.layers]
        for i, weight_list in enumerate(initial_weights):
            for j, weight_matrix in enumerate(weight_list):
                assert np.mean(np.abs(weight_matrix - final_weights[i][j])) > 0

    # tests a patch based model on all volumes, saves the results in a .csv file
    def test_at_split(self, split_dict, folder_split):
        ids_train = split_dict['train']
        ids_valid = split_dict['validation']
        ids_test = split_dict['test']
        all_ids = sorted(ids_train + ids_valid + ids_test)
        num_vols = len(all_ids)

        metrics = ['MSE_NBG', 'MSE', 'SSIM_NBG', 'PSNR_NBG', 'SSIM', 'PSNR', 'MSE_NBG_AVG_EMB']

        print('testing model on all volumes...')

        # create files
        files_embs = {}
        for emb in range(self.mm.num_emb):
            files = {}
            for mod in self.output_modalities:
                csv_header = '#,' + ','.join(metrics[:-1]) + ', volume_type, MSE_NBG_AVG_EMB\n'
                csv_file = folder_split + '/individual_results_emb_' + str(emb) + '_mod_' + mod + '.csv'
                fd = open(csv_file, "w")
                fd.write(csv_header)
                files[mod] = fd
            files_embs[emb] = files
        print 'Created ' + str(len(files_embs)) + ' test files'

        if not os.path.exists(folder_split + '/avg_emb_ims'):
            os.makedirs(folder_split + '/avg_emb_ims')

        for vol_num in range(num_vols):
            if vol_num not in ids_test:
                continue

            print('testing model on volume ' + str(vol_num) + '...')

            X = [self.data.select_for_ids(mod, [vol_num]) for mod in self.input_modalities]
            Z = self.mm.model.predict(X)

            # compute emb average
            err_avg_emb = self.output_mean(Z, folder_split, vol_num)

            for emb in range(self.mm.num_emb):
                files = files_embs[emb]
                for yi, y in enumerate(self.output_modalities):
                    z_idx = [outi for outi, out in enumerate(self.mm.model.outputs) if
                             self.output_modalities[yi] in out.name]
                    y_synth = [Z[zi][:, 0] for zi in z_idx]

                    y_truth = self.data.select_for_ids(y, [vol_num])

                    if y_truth.shape[1] > 1:
                        sh = y_truth.shape
                        y_truth = y_truth[:, sh[1] / 2:sh[1] / 2 + 1]

                    err = ErrorMetrics(y_synth[emb], y_truth)

                    vol_type = ''
                    if vol_num in ids_test:
                        vol_type = 'test'
                    if vol_num in ids_valid:
                        vol_type = 'validation'
                    if vol_num in ids_train:
                        vol_type = 'training'

                    pattern = "%d" + ", %.3f" * (len(metrics) - 1) + ', %s, %.3f\n'
                    new_row = pattern % tuple([vol_num] + list([err[em] for em in metrics[:-1]]) + [vol_type] +
                                              [err_avg_emb[y]['MSE_NBG']])
                    files[y].write(new_row)

        for files in files_embs.values():
            for fd in files.values():
                fd.close()

        cb_X = [self.data.select_for_ids(mod, all_ids, as_array=False) for mod in self.input_modalities]
        cb_Y = [self.data.select_for_ids(mod, all_ids, as_array=False) for mod in self.output_modalities]
        cb = ImageSaveCallback(cb_X, cb_Y, None, None, folder_split, self.output_modalities)
        cb.model = self.mm.model
        for vol in ids_test + [1, 8, 12, 13]:
            cb.saveImage(vol, [70, 80, 90, 100], folder_split + '/test_im' + str(vol), cb_X, cb_Y)

    def output_mean(self, z_vol, folder_split, vol_num):
        err_avg_emb = dict()
        for yi, y in enumerate(self.output_modalities):
            z_idx = [outi for outi, out in enumerate(self.mm.model.outputs) if self.output_modalities[yi] in out.name]
            y_synth = [z_vol[zi][:, 0] for zi in z_idx]

            y_truth_mod = self.data.select_for_ids(y, [vol_num])
            if y_truth_mod.shape[1] > 1:
                sh = y_truth_mod.shape
                y_truth_mod = y_truth_mod[:, sh[1] / 2:sh[1] / 2 + 1]

            y_synth_mod = np.mean(y_synth, axis=0)
            scipy.misc.imsave(folder_split + '/avg_emb_ims/im_avg_emb' + str(vol_num) + '_' + str(90) + '.png',
                              np.concatenate([y_synth_mod[90], y_truth_mod[90, 0]], axis=1))
            err = ErrorMetrics(np.expand_dims(y_synth_mod, axis=1), y_truth_mod)
            err_avg_emb[y] = err
        return err_avg_emb
