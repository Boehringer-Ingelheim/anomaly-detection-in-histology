import torch
import numpy as np
import logging
import torch.nn as nn  # must stay here
import torch.optim as optim  # must stay here

import os


def apply_net(model, device, val_data_loader, n_val_samples=np.inf, verbose=True, code_processor=None, image_processor=None):

    """
    Validation of the trained model

    :param device: device to put data on
    :param val_data_loader: data loader that provides images to be processed
    :param model: network model to be used. It should already be on the right device
    :param n_val_samples: if given uses that number of samples from data loader
    :param code_processor: callable class that can gather code vectors
    :param image_processor: receives input and output images (of auto-encoder) and computes and saves a reconstruction error

    """

    def get_patch_coordinates(samples):
        # reads 'info' field of samples with stored x, y, w, h information in the string and parse it to dictionary
        # with the corresponding x, y, w, h fields, each of which contains a list of integers of the length equal
        # to the number of images in the batch

        if 'info' in samples.keys():
            patch_info = samples['info']
        else:
            return None

        x, y, w, h = [], [], [], []
        for info in patch_info:

            temp = info.split(', ')[0].split(': ')
            assert temp[0] == 'x'
            x.append(int(temp[1]))

            temp = info.split(', ')[1].split(': ')
            assert temp[0] == 'y'
            y.append(int(temp[1]))

            temp = info.split(', ')[2].split(': ')
            assert temp[0] == 'w'
            w.append(int(temp[1]))

            temp = info.split(', ')[3].split(': ')
            assert temp[0] == 'h'
            h.append(int(temp[1]))

        return {'x': x, 'y': y, 'w': w, 'h': h}


    def extend_image_names(samples):

        if 'info' in samples.keys():
            patch_info = samples['info']
        else:
            patch_info = ['x: N, y: N, w: N, h: N'] * len(samples['image_name'])

        im_names = []
        for im_path, info in zip(samples['image_name'], patch_info):
            info = info.split(',')
            x = info[0].split()[1]
            y = info[1].split()[1]
            w = info[2].split()[1]
            h = info[3].split()[1]

            image_name = os.path.basename(im_path)
            image_name = os.path.splitext(image_name)[0]
            im_names.append(image_name + '_x' + x + '_y' + y + '_w' + w + '_h' + h)

        return im_names

    init_model_status = model.training
    model.eval()

    val_loss = torch.tensor(0.0)
    diff_measure = []
    n_processed = 0
    data_iter = iter(val_data_loader)
    with torch.no_grad():
        counter = 0

        while n_processed < n_val_samples:

            try:
                samples = next(data_iter)
            except StopIteration:
                if n_val_samples != np.inf:
                    logging.warning(
                        'Warning: finite iterator was provided, it was exhausted before required n_samples were generated')
                break

            images = samples['image'].to(device)

            output_dict = model(images)

            if 'rec_image' in output_dict:
                outputs = output_dict['rec_image']
            else:
                outputs = None

            if 'codes' in output_dict:
                embeddings = output_dict['codes']
            elif 'pooled_codes' in output_dict:
                embeddings = output_dict['pooled_codes']
            else:
                embeddings = None

            n_processed += images.shape[0]
            counter += 1

            # collect computed codes if requested
            if code_processor is not None:

                im_names = samples['image_name'][:]
                im_labels = samples['string_label'][:]
                coordinates = get_patch_coordinates(samples)

                code_processor(embeddings, im_names, im_labels, coordinates)



            if image_processor is not None:
                if image_processor.save_images_path:

                    im_names = extend_image_names(samples)
                    image_processor(images, outputs, im_names)

                else:
                    image_processor(images, outputs)


    if verbose:
        logging.info('{} validation patches were read'.format(n_processed))

    # setting initial status of the model
    model.train(init_model_status)

    return None









