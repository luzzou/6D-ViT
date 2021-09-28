# 6D-ViT

![teaser](demo/fig_visualization_real.png)




## Overview

In this repo, we provide our pose estimation results on the REAL275 test set to evaluate the performance of our method.

More implementation details will be released once the paper is accepted.





## Dependencies

* Python 3.6
* PyTorch 1.7.1+cu110
* CUDA 11.2



## Evaluation on the REAL275 testset


1. Download the Mask R-CNN results,  pose predictions by NOCS, NOF, SPD and our 6D-ViT from [here](https://drive.google.com/drive/folders/1nfELPlLWQwbGd4U5rC-l6wll7dkE4DEL)

2. Download the testlist from [here](https://drive.google.com/drive/folders/1nfELPlLWQwbGd4U5rC-l6wll7dkE4DEL)


```
unzip -q real_test.zip
mv real_test/ $ROOT/results
mv test_list.txt $ROOT/
cd $ROOT
python evaluate_mean_real.py
```



# Acknowledgement

We thank for [object-deformnet](https://github.com/mentian/object-deformnet) for releasing their code.
