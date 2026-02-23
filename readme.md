WELCOME to gpufish

This is a package inspired by fishquant/bigfish, please refer to their website. 
Our paper is in preparation, I will post it when we have a pre-print. 

1. What is gpufish? 
gpufish, is a pacakge developed for anyone dont want to waste time on coding their own pipeline of smFISH spot detection and want to run smFISH detection on any GPU platform. This code is mostly compatible with cupy and cucim, if you are not using gpu please turn to fish-quant or other platforms. 

2. What data should use gpufish? 
gpufish is designed for large datasets, which might require high amount of RAM to run. You need a cuda-ready GPU, and 3D images that are big and taks a very long time to processon CPU. Dont worry if you have a really bad signal to noise ratio, as long as you can visuallize your spots then gpufish can help. 

3. What does gpufish do? 
This is as simple as almost all smFISH processing package. Log filter - local maximum filter - spot size filter (if needed) - weighted region properties (will be explained later). There is no compelx part, except for some sanity checks and utility functions, we use scipy ndimage log filter, local maximum filter, and spot size just based on size, and skimage regionprops to get spots. 

4. What is special about gpufish? 
I added a filter based on not only intensity, but other regionprops. We are mostly asking a question: under what threshold we can treat a signal as background? So I added a function to fully check different parameters on different regionprops, you can define any regionprops you like, and I also added SNR, center intensity/mean intensity, and exceeding, center intenstiy - mean intensity, to find spots that really have a center intensity higher than spots around. This helps images on tissues where background is very high. 

