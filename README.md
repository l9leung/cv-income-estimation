## Computer Vision Income Estimation

Estimating median household income using Google Street View.

* `data/streetview_data.py` obtains up to 10 Google Street View images per census block group from the given city
* `data/acs_data.py` obtains the median household income by census block group from the given city
* `vgg16.py` extracts image features using VGG16 ConvNet
* `models.py` fits a $\varepsilon$-SVR model on half the images and tests out-of-sample $R^2$ on the other half

![Alt text](https://raw.githubusercontent.com/l9leung/cv-income-estimation/main/data/visualizations/income_choro_Los%20Angeles.png)
