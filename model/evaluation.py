# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import SimpleITK as sitk
import torch as th

import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import airlab as al


def affine_reg(img_draw, img_ref, output_path, lr=0.01, iter=200):
    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    device = th.device("cpu")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    # device = th.device("cuda:0")

    # load the image data and normalize to [0, 1]
    itkImg = sitk.ReadImage(img_ref, sitk.sitkFloat32)
    itkImg = sitk.RescaleIntensity(itkImg, 0, 1)
    fixed_image = al.create_tensor_image_from_itk_image(itkImg, dtype=dtype, device=device)
    fsize = fixed_image.numpy().flatten().size

    itkImg = sitk.ReadImage(img_draw, sitk.sitkFloat32)
    itkImg = sitk.RescaleIntensity(itkImg, 0, 1)
    moving_image = al.create_tensor_image_from_itk_image(itkImg, dtype=dtype, device=device)


    # create pairwise registration object
    registration = al.PairwiseRegistration(dtype=dtype, device=device)

    # choose the affine transformation model
    transformation = al.transformation.pairwise.RigidTransformation(moving_image.size, dtype=dtype, device=device)
    registration.set_transformation(transformation)

    # choose the scaling regulariser and the diffusion regulariser
    # scale_reg = al.regulariser.parameter.ScalingRegulariser('trans_parameters')
    # scale_reg.set_weight(0.0001)
    # registration.set_regulariser_parameter([scale_reg])
    # dis_reg = al.regulariser.displacement.DiffusionRegulariser(moving_image.spacing, size_average=False)
    # registration.set_regulariser_displacement([dis_reg])
    # dis_reg.set_weight(0.1)


    # choose the Mean Squared Error as image loss
    image_loss = al.loss.pairwise.NCC(fixed_image, moving_image)
    #init_loss = np.sum(np.square(fixed_image.numpy() - moving_image.numpy()))/fsize
    registration.set_image_loss([image_loss])

    # choose the Adam optimizer to minimize the objective
    optimizer = th.optim.Adam(transformation.parameters(), lr=lr)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(iter)

    # start the registration
    registration.start()

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image, displacement)

    param = transformation.trans_parameters.detach().numpy()
    translate = np.sqrt(np.square(param[1]) + np.square(param[2]))
    scale = np.sqrt( ( np.square(param[3]-1) + np.square(param[4] - 1) ) * 0.5  )
    init_loss = registration.init_loss.detach().numpy()
    final_loss = registration.img_loss.detach().numpy()


    # plot the results
    # plt.subplot(131)
    # plt.imshow(fixed_image.numpy(), cmap='gray')
    # plt.title('Ref')

    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.xticks([])
    plt.yticks([])
    fixed = fixed_image.numpy()
    moved = moving_image.numpy()
    warp = warped_image.numpy()
    csfont = {'fontname': 'Times New Roman', 'size':35}

    p1 = np.ones((fixed.shape[0], fixed.shape[1], 3))
    p1[fixed < 1] = [0.5, 0.5, 0.5]
    p1[moved < 1] = [0.4, 0.62, 0.78]
    plt.imshow(p1)
    plt.title('Raw', **csfont)

    plt.subplot(122)
    plt.xticks([])
    plt.yticks([])
    p2 = np.ones((fixed.shape[0], fixed.shape[1], 3))
    p2[fixed < 1] = [0.5, 0.5, 0.5]
    p2[warp < 0.5] = [0.4, 0.62, 0.78]
    plt.imshow(p2)
    plt.title('Transformed', **csfont)

    plt.savefig(output_path, bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    return init_loss, final_loss, np.abs(param[0]), translate, scale, warped_image
