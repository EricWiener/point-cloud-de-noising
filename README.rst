==============
pcd-de-noising
==============


PyTorch Lightning implementation of CNN-Based Lidar Point Cloud De-Noising in Adverse Weather. The
original paper can be found on `arvix`_. The data used in the paper is available in the `PointCloudDeNoising repository`_.

Documentation and contributing guidelines can be found on `readthedocs`_.

Quick Start
===========
Create a Conda enviroment::

   conda create -n pcd-de-noising python=3 h5py

Then activate the environment ``pcd-de-noising`` with::

   conda activate pcd-de-noising
   conda install -c conda-forge pyscaffold tox pytorch-lightning
   conda install pytorch -c pytorch

Then you can run the ``train.ipynb`` notebook to quickly train, validate, and run inference. It is all setup with checkpoint loading and tensorboard logging.


Paper Abstract
==============

Lidar sensors are frequently used in environment perception for autonomous vehicles and mobile robotics to complement camera, radar, and ultrasonic sensors. Adverse weather conditions are significantly impacting the performance of lidar-based scene understanding by causing undesired measurement points that in turn effect missing detections and false positives. In heavy rain or dense fog, water drops could be misinterpreted as objects in front of the vehicle which brings a mobile robot to a full stop. In this paper, we present the first CNN-based approach to understand and filter out such adverse weather effects in point cloud data. Using a large data set obtained in controlled weather environments, we demonstrate a significant performance improvement of our method over state-of-the-art involving geometric filtering.

Making Changes & Contributing
=============================

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd pcd-de-noising
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate

Note
====

This project has been set up using PyScaffold 4.0.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

.. _arvix: https://arxiv.org/abs/1912.03874
.. _PointCloudDeNoising repository: https://github.com/rheinzler/PointCloudDeNoising
.. _readthedocs: https://point-cloud-de-noising.readthedocs.io/en/latest/
