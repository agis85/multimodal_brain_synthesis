# Multimodal Brain Synthesis

This project performs multimodal MR brain synthesis using modality invariant latent representations. For details see
our papers [Multimodal MR Synthesis via Modality-Invariant Latent Representation] and [Robust Multi-Modal MR Image Synthesis].

The main files in this project are:

* model.py: contains the neural network implementation
* loader_multimodal.py: loads the input data into a Data object and performs pre-processing.
* runner.py: creates an Experiment object to perform cross validation on a given Data object.

An example, assuming usage of a Data object is the following:

```
data = Data(dataset='ISLES', trim_and_downsample=False)
data.load()

input_modalities = ['T1', 'T2', 'DWI']
output_weights = {'VFlair': 1.0, 'concat': 1.0}
exp = Experiment(input_modalities, output_weights, '/path/to/foldername', data, latent_dim=16, spatial_transformer=True)
exp.run(data)
```

## Data

The sources used to evaluate our methods are brain data from [ISLES] and [BRATS] challenges, as well as the [IXI] dataset.

## Citation

If you use this code for your research, please cite our papers:

```
@article{chartsias2017multimodal,
  title={Multimodal MR Synthesis via Modality-Invariant Latent Representation},
  author={Chartsias, Agisilaos and Joyce, Thomas and Giuffrida, Mario Valerio and Tsaftaris, Sotirios A},
  journal={IEEE Transactions on Medical Imaging},
  year={2017},
  publisher={IEEE}
}

@inproceedings{joyce2017robust,
  title={Robust Multi-modal MR Image Synthesis},
  author={Joyce, Thomas and Chartsias, Agisilaos and Tsaftaris, Sotirios A},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={347--355},
  year={2017},
  organization={Springer}
}
```

## Acknowledgements

The project uses a [Spatial Transformer] implementation, distributed under MIT licence.
 
[Multimodal MR Synthesis via Modality-Invariant Latent Representation]: http://ieeexplore.ieee.org/document/8071026/
[Robust Multi-Modal MR Image Synthesis]: https://link.springer.com/chapter/10.1007/978-3-319-66179-7_40
[Spatial Transformer]: https://github.com/skaae/transformer_network
[ISLES]: http://www.isles-challenge.org/ISLES2015/
[BRATS]: https://sites.google.com/site/braintumorsegmentation/home/brats2015
[IXI]: http://brain-development.org/ixi-dataset/
