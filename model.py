
import numpy as np
import matplotlib

matplotlib.use('Agg')
import sys

sys.setrecursionlimit(10000)

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, merge, Lambda, LeakyReLU, MaxPooling2D
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras import backend as K
from keras.layers.core import Dense, Activation, Flatten

from SpatialTransformerLayer import SpatialTransformer


class Multimodel(object):
    '''
    Class for constructing a neural network model as described in
    T. Joyce, A. Chartsias, S.A. Tsaftaris, 'Robust Multi-Modal MR Image Synthesis,' MICCAI 2017

    The compiled Keras model inputs and outputs are the following:
    inputs: list of numpy data arrays, one for each modality
    outputs: list containing numpy arrays for each output modality, 2 zero numpy arrays (one used for variance minimisation,
    the other as a dummy value since the last output of the model contains latent representations)

    The model is 2D, so the input numpy arrays are of size (<num_images>, <channels>, <height>, <width>)

    Example usage:
    m = Multimodel(['T1','T2'], ['DWI', 'VFlair'], {'DWI': 1.0, 'VFlair': 1.0, 'concat': 1.0}, 16, 1, True, 'max', True, True)
    m.build()
    '''
    def __init__(self, input_modalities, output_modalities, output_weights, latent_dim, channels, spatial_transformer,
                 common_merge, ind_outs, fuse_outs):
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities
        self.latent_dim = latent_dim
        self.channels = channels
        self.common_merge = common_merge
        self.output_weights = output_weights
        self.spatial_transformer = spatial_transformer
        self.ind_outs = ind_outs
        self.fuse_outs = fuse_outs
        self.num_emb = len(input_modalities) + 1

        if spatial_transformer:
            self.H, self.W = 112, 80  # Width/Height for ISLES2015 dataset
        else:
            self.H, self.W = None, None

    def encoder_maker(self, modality):
        inp = Input(shape=(self.channels, self.H, self.W), name='enc_' + modality + '_input')
        conv = Conv2D(32, 3, padding='same', name='enc_' + modality + '_conv1')(inp)
        act = LeakyReLU()(conv)
        conv = Conv2D(32, 3, padding='same', name='enc_' + modality + '_conv2')(act)
        act1 = LeakyReLU()(conv)

        # downsample 1st level
        pool = MaxPooling2D(pool_size=(2, 2))(act1)
        conv = Conv2D(64, 3, padding='same', name='enc_' + modality + '_conv3')(pool)
        act = LeakyReLU()(conv)
        conv = Conv2D(64, 3, padding='same', name='enc_' + modality + '_conv4')(act)
        act2 = LeakyReLU()(conv)

        # downsample 2nd level
        pool = MaxPooling2D(pool_size=(2, 2))(act2)
        conv = Conv2D(128, 3, padding='same', name='enc_' + modality + '_conv5')(pool)
        act = LeakyReLU()(conv)
        conv = Conv2D(128, 3, padding='same', name='enc_' + modality + '_conv6')(act)
        act = LeakyReLU()(conv)

        # upsample 2nd level
        ups = UpSampling2D(size=(2, 2))(act)
        conv = Conv2D(64, 3, padding='same', name='enc_' + modality + '_conv7')(ups)
        skip = merge([act2, conv], mode='concat', concat_axis=1, name='enc_' + modality + '_skip1')
        conv = Conv2D(64, 3, padding='same', name='enc_' + modality + '_conv8')(skip)
        act = LeakyReLU()(conv)
        conv = Conv2D(64, 3, padding='same', name='enc_' + modality + '_conv9')(act)
        act = LeakyReLU()(conv)

        # upsample 2nd level
        ups = UpSampling2D(size=(2, 2))(act)
        conv = Conv2D(32, 3, padding='same', name='enc_' + modality + '_conv10')(ups)
        skip = merge([act1, conv], mode='concat', concat_axis=1, name='enc_' + modality + '_skip2')
        conv = Conv2D(32, 3, padding='same', name='enc_' + modality + '_conv11')(skip)
        act = LeakyReLU()(conv)
        conv = Conv2D(32, 3, padding='same', name='enc_' + modality + '_conv12')(act)
        act = LeakyReLU()(conv)

        conv_ld = self.latent_dim / 2 if self.common_merge == 'hemis' else self.latent_dim
        conv = Conv2D(conv_ld, 3, padding='same', name='enc_' + modality + '_conv13')(act)
        lr = LeakyReLU()(conv)

        return inp, lr

    def decoder_maker(self, modality):
        inp = Input(shape=(self.latent_dim, None, None), name='dec_' + modality + '_input')
        conv = Conv2D(32, 3, padding='same', activation='relu', name='dec_' + modality + '_conv1')(inp)
        conv = Conv2D(32, 3, padding='same', activation='relu', name='dec_' + modality + '_conv2')(conv)
        skip = merge([inp, conv], mode='concat', concat_axis=1, name='dec_' + modality + '_skip1')
        conv = Conv2D(32, 3, padding='same', activation='relu', name='dec_' + modality + '_conv3')(skip)
        conv = Conv2D(32, 3, padding='same', activation='relu', name='dec_' + modality + '_conv4')(conv)
        skip = merge([skip, conv], mode='concat', concat_axis=1, name='dec_' + modality + '_skip2')
        conv = Conv2D(1, 1, padding='same', activation='relu', name='dec_' + modality + '_conv5')(skip)
        model = Model(input=inp, output=conv, name='decoder_' + modality)
        return model

    def get_embedding_distance_outputs(self, embeddings):
        if len(self.inputs) == 1:
            print 'Skipping embedding distance outputs for unimodal model'
            return []

        outputs = list()

        ind_emb = embeddings[:-1]
        weighted_rep = embeddings[-1]

        all_emb_flattened = [new_flatten(emb) for emb in ind_emb]
        concat_emb = merge(all_emb_flattened, mode='concat', concat_axis=1, name='em_concat')
        concat_emb.name = 'em_concat'

        outputs.append(concat_emb)
        print 'making output: em_concat', concat_emb.type, concat_emb.name

        fused_emb = new_flatten(weighted_rep, name='em_fused')
        fused_emb.name = 'em_fused'
        outputs.append(fused_emb)

        return outputs

    # HeMIS based fusion:
    # M. Havaei, N. Guizard, N. Chapados, and Y. Bengio, “HeMIS: Hetero- modal image segmentation,”
    # in MICCAI. Springer, 2016, pp. 469–477
    def hemis(self, ind_emb):
        if len(self.input_modalities) == 1:
            combined_emb1 = ind_emb[0]
            combined_emb2 = K.zeros_like(ind_emb[0])  # if we only have one input the variance is 0
        else:
            combined_emb1 = merge(ind_emb, mode='ave', name='combined_em_ave',
                                  output_shape=(self.latent_dim / 2, None, None))
            combined_emb2 = merge(ind_emb, mode=var, name='combined_em_var',
                                  output_shape=(self.latent_dim / 2, None, None))

        combined_emb = merge([combined_emb1, combined_emb2], mode='concat', concat_axis=1, name='combined_em',
                             output_shape=(self.latent_dim, None, None))

        new_ind_emb = []
        for i, emb in enumerate(ind_emb):
            new_ind_emb.append(merge([emb, zeros_for_var(emb)], mode='concat', concat_axis=1, name='emb_' + str(i),
                                     output_shape=(self.latent_dim, None, None)))
        ind_emb = new_ind_emb

        all_emb = ind_emb + [combined_emb]
        return all_emb

    def build(self):
        print 'Latent dimensions: ' + str(self.latent_dim)

        encoders = [self.encoder_maker(m) for m in self.input_modalities]

        ind_emb = [lr for (input, lr) in encoders]
        self.org_ind_emb = [lr for (input, lr) in encoders]
        self.inputs = [input for (input, lr) in encoders]

        # apply spatial transformer
        if self.spatial_transformer:
            print 'Adding a spatial transformer layer'
            input_shape = (self.latent_dim, self.H, self.W)
            tpn = tpn_maker(input_shape)
            mod1 = ind_emb[0]
            aligned_ind_emb = [mod1]
            for mod in ind_emb[1:]:
                aligned_mod = merge([tpn([mod1, mod]), mod], mode=STMerge, output_shape=input_shape)
                aligned_ind_emb.append(aligned_mod)
            ind_emb = aligned_ind_emb

        if self.common_merge == 'hemis':
            self.all_emb = self.hemis(ind_emb)
        else:
            assert self.common_merge == 'max' or self.common_merge == 'ave' or self.common_merge == 'rev_loss'
            print 'Fuse latent representations using ' + str(self.common_merge)
            cm = 'max' if self.common_merge == 'rev_loss' else self.common_merge
            weighted_rep = merge(ind_emb, mode=cm, name='combined_em') if len(self.inputs) > 1 else ind_emb[0]
            self.all_emb = ind_emb + [weighted_rep]

        self.decoders = [self.decoder_maker(m) for m in self.output_modalities]
        outputs = get_decoder_outputs(self.output_modalities, self.decoders, self.all_emb)

        # this is for minimizing the distance between the individual embeddings
        outputs += self.get_embedding_distance_outputs(self.all_emb)

        print 'all outputs: ', [o.name for o in outputs]

        out_dict = {'em_%d_dec_%s' % (emi, dec): mae for emi in range(self.num_emb) for dec in self.output_modalities}

        get_indiv_weight = lambda mod: self.output_weights[mod] if self.ind_outs else 0.0
        get_fused_weight = lambda mod: self.output_weights[mod] if self.fuse_outs else 0.0
        loss_weights = {}
        for dec in self.output_modalities:
            for emi in range(self.num_emb - 1):
                loss_weights['em_%d_dec_%s' % (emi, dec)] = get_indiv_weight(dec)
            loss_weights['em_%d_dec_%s' % (self.num_emb - 1, dec)] = get_fused_weight(dec)

        if len(self.inputs) > 1:
            if self.common_merge == 'rev_loss':
                out_dict['em_concat'] = mae
            else:
                out_dict['em_concat'] = embedding_distance
            loss_weights['em_concat'] = self.output_weights['concat']

            out_dict['em_fused'] = embedding_distance
            loss_weights['em_fused'] = 0.0

        print 'output dict: ', out_dict
        print 'loss weights: ', loss_weights

        self.model = Model(input=self.inputs, output=outputs)
        self.model.compile(optimizer=Adam(lr=0.0001), loss=out_dict, loss_weights=loss_weights)

    def get_inputs(self, modalities):
        return [self.inputs[self.input_modalities.index(mod)] for mod in modalities]

    def get_embeddings(self, modalities):
        assert set(modalities).issubset(set(self.input_modalities))
        ind_emb = [self.all_emb[self.input_modalities.index(mod)] for mod in modalities]
        org_ind_emb = [self.org_ind_emb[self.input_modalities.index(mod)] for mod in modalities]

        if self.common_merge == 'hemis':
            combined_emb1 = merge(org_ind_emb, mode='ave', name='combined_em_ave',
                                  output_shape=(self.latent_dim / 2, None, None))
            combined_emb2 = merge(org_ind_emb, mode=var, name='combined_em_var',
                                  output_shape=(self.latent_dim / 2, None, None))

            combined_emb = merge([combined_emb1, combined_emb2], mode='concat', concat_axis=1, name='combined_em',
                                 output_shape=(self.latent_dim, None, None))

            new_ind_emb = []
            for i, emb in enumerate(org_ind_emb):
                new_ind_emb.append(merge([emb, zeros_for_var(emb)], mode='concat', concat_axis=1, name='pemb_' + str(i),
                                         output_shape=(self.latent_dim, None, None)))
            ind_emb = new_ind_emb

            return ind_emb + [combined_emb]
        else:
            if len(ind_emb) > 1:
                fused_emb = merge(ind_emb, mode=self.common_merge, name='fused_em')
            else:
                fused_emb = ind_emb[0]
            return ind_emb + [fused_emb]

    def get_input(self, modality):
        assert modality in self.input_modalities
        for l in self.model.layers:
            if l.name == 'enc_' + modality + '_input':
                return l.output
        return None

    def predict_z(self, input_modalities, data, ids):
        embeddings = self.get_embeddings(input_modalities)
        inputs = [self.get_input(mod) for mod in input_modalities]
        partial_model = Model(input=inputs, output=embeddings)
        X = [data.select_for_ids(inmod, ids) for inmod in input_modalities]
        Z = partial_model.predict(X)
        assert len(Z) == len(embeddings)
        return Z

    def new_decoder_model(self, input_modalities, modality):
        if modality in self.output_modalities:
            print 'Using trained decoder'
            decoder = self.decoders[self.output_modalities.index(modality)]
        else:
            print 'Creating new decoder'
            decoder = self.decoder_maker(modality)
        inputs = [Input(shape=(self.latent_dim, None, None)) for i in range(len(input_modalities) + 1)]

        outputs = [decoder(inpt) for inpt in inputs]
        for outi, out in enumerate(outputs):
            out.name = 'em_%d_dec_%s' % (outi, modality)

        out_dict = {decoder.name: mae}
        loss_weights = {decoder.name: 1.0}

        new_model = Model(input=inputs, output=outputs)
        new_model.compile(optimizer=Adam(lr=0.0001), loss=out_dict, loss_weights=loss_weights)

        return new_model

    def get_partial_model(self, input_modalities, output_modality):
        assert set(input_modalities).issubset(set(self.input_modalities))
        assert output_modality in self.output_modalities

        inputs = self.get_inputs(input_modalities)
        embeddings = self.get_embeddings(input_modalities)

        decoder = self.decoders[self.output_modalities.index(output_modality)]
        outputs = get_decoder_outputs([output_modality], [decoder], embeddings)
        outputs += self.get_embedding_distance_outputs(embeddings)

        model = Model(input=inputs, output=outputs)
        return model

    def new_encoder_model(self, modality, output_modalities):
        if modality in self.input_modalities:
            print 'Using trained encoder'
            input = self.inputs[self.input_modalities.index(modality)]
            lr = self.all_emb[self.input_modalities.index(modality)]
        else:
            print 'Creating new encoder'
            input, lr = self.encoder_maker(modality)

        decoders = [self.decoders[self.output_modalities.index(mod)] for mod in output_modalities]
        for d in decoders:
            d.trainable = False
        outputs = get_decoder_outputs(output_modalities, decoders, [lr])

        model = Model(input=[input], output=outputs)
        model.compile(optimizer=Adam(), loss={d.name: mae for d in decoders},
                      loss_weights={d.name: 1.0 for d in decoders})
        return model

def get_decoder_outputs(output_modalities, decoders, embeddings):
    assert len(output_modalities) == len(decoders)

    outputs = list()
    for di, decode in enumerate(decoders):
        for emi, em in enumerate(embeddings):
            out_em = decode(em)
            name = 'em_' + str(emi) + '_dec_' + output_modalities[di]
            l = Lambda(lambda x: x + 0, name=name)(out_em)
            l.name = name
            outputs.append(l)
            print 'making output:', em.type, out_em.type, name

    return outputs

def embedding_distance(y_true, y_pred):
    return K.var(y_pred, axis=1)


def new_flatten(emb, name=''):
    l = Lambda(lambda x: K.batch_flatten(x))(emb)
    l = Lambda(lambda x: K.expand_dims(x, axis=1), name=name)(l)
    return l


def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def var(embeddings):
    emb = embeddings[0]
    shape = (emb.shape[1], emb.shape[2], emb.shape[3])
    sz = shape[0] * shape[1] * shape[2]

    flat_embs = [K.reshape(emb, (emb.shape[0], 1, sz)) for emb in embeddings]

    emb_var = K.var(K.concatenate(flat_embs, axis=1), axis=1, keepdims=True)

    return K.reshape(emb_var, embeddings[0].shape)


def zeros_for_var(emb):
    l = Lambda(lambda x: K.zeros_like(x))(emb)
    return l


def STMerge(to_merge):
    theta, input = to_merge
    theta.reshape((input.shape[0], 2, 3))
    return SpatialTransformer._transform(theta, input, 1)


def tpn_maker(input_shape):
    # initial weights
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]

    # input_shape = (1, 112, 80)

    target_input = Input(shape=input_shape)
    input = Input(shape=input_shape)

    stacked = merge([target_input, input], mode='concat')

    mp1 = MaxPooling2D(pool_size=(2, 2))(stacked)
    conv1 = Conv2D(8, 5)(mp1)
    mp2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(8, 5)(mp2)
    mp3 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(8, 5)(mp3)
    flt = Flatten()(conv3)
    d50 = Dense(50)(flt)
    act = Activation('relu')(d50)
    theta = Dense(6, weights=weights)(act)

    model = Model(input=[target_input, input], output=theta)
    return model
