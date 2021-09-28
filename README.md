# 6D-ViT

![teaser](demo/fig_visualization_real.png)




## Overview

In this repo, we provide our pose estimation results on the REAL275 test set to evaluate the performance of our method.

More implementation details about our work will be released once the paper is accepted.





## Dependencies

* Python 3.6
* PyTorch 1.7.1+cu110
* CUDA 11.2



## Evaluation on the REAL275 test set



Download the segmentation results from Mask R-CNN, predictions by NOCS, NOF, SPD and our results from [here](https://drive.google.com/drive/folders/1nfELPlLWQwbGd4U5rC-l6wll7dkE4DEL)



```
unzip -q real_test.zip
mv real_test/ $ROOT/results
cd $ROOT
python evaluate.py
```



# Acknowledgement

We thank for [object-deformnet](https://github.com/mentian/object-deformnet) for releasing their code.
