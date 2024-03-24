
# Introduction
## Motivation
Accurately quantifying cellular morphology at scale could significantly enhance cell analysis techniques. However, essential to acknowledge that data handling and preparation is a foundation of any large-scale AI processing. This leads us to confront two primary challenges: managing the voluminous data and maximizing the utility of the information it encapsulates. Your contributions could significantly improve our processing methods and potentially have a considerable impact on the quality of our analysis.

## Background
Cell painting images are characterized by their multi-channel nature, extending beyond the typical three-channel (RGB) format. Each channel in cell painting images represents a different cellular structure, such as mitochondria. Conventional data reduction methods, like converting files to JPG, are dedicated for three-channel images, thus making them not suitable. Moreover, many unsupervised learning algorithms involve extracting a patch from the original image and comparing it to a corresponding positive sample. For example, DINO model creates two views of the same image: one global view for the "teacher" and one local view for the "student," with each view being randomly selected. These views are then compared, with the model learning the similarities between them. However, there can be instances where views do not contain any cells, potentially harming the model's performance.

## Task
Task:
You will receive a data frame with image IDs and coordinates for cell bounding boxes (one to many relationship). Your task is to generate patches that maximize coverage of the area within the original bounding boxes, while adhering to a predetermined size (e.g., 256x256 – but should be parameterized). Importantly, the collective area of these new patches must not surpass that of the original image (1080x1080), aiming to decrease the total data size. Additionally, these patches should minimize background exposure and avoid overlapping. 

There are three primary metrics we want to focus on:
1.	Global Coverage Percentage: We aim to see an improvement in the histogram shift to the right, indicating your solution covers more of the original bounding boxes than our approach.
2.	Local Coverage Percentage: Similarly, we desire a histogram shift to the right, showing your patches incorporate less background than ours.
3.	Total Area of New Patches: The smaller, the better, ensuring it remains beneath the aggregate area of the images (1080x1080 multiplied by the number of images).

Most probably the higher the global coverage the lower the local coverage – we are aware of it. However, we did not yet come up with a single metric that encapsulates all the above – we encourage you to propose your own.

## Requirements
1.	Processing Time: It should be efficient given our extensive dataset. Our current solution takes approx. 3 minutes to complete. Anything below an hour will be fine. We value actual time savings over computational complexity theories, such as Big O notation. Solutions employing multiprocessing, vectorization, or other optimizations for time efficiency will be looked upon favorably.
2.	The solution must be implemented in Python, aligning with our existing codebase infrastructure.

# User guide
This guide provides rules and examples on how to get started with 2d_bin_packing!

## Setting up project

Some of the libraries used do not support newer versions of python. Therefore one have to create
virtual environment with specified version of python. There are several ways to do this, e.g. using conda:

```conda create --name 2d_bin_packing python=3.10.13```

Having created the environment we have to active it:

```conda activate 2d_bin_packing```

Subsequently one have to install poetry:

```conda install -c conda-forge poetry==1.5.1```

Last step is to install libraries using poetry:

```poetry install```



## Adding new packages

Adding new packages is as simple as ```poetry add {package_name}```. However, poetry introduces nice feature called 
[dependecy groups](https://python-poetry.org/docs/master/managing-dependencies/). In order to add packages as a part 
of a group, all you need to do is ```poetry add --group {group_name} {package_name}```.