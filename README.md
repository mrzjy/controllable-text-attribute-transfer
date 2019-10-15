The official code is written in Pytorch (Please refer to <https://github.com/Nrgeup/controllable-text-attribute-transfer> to see the original code as well as the paper), 
This repo reimplements it in Tensorflow for some others' interest. And since I cannot play with gradients at ease with estimator api, I just implemented this in the old-fashion way (sess.run(feed)).

## Dependencies
~~~
Tensorflow 1.11.0 (or above)
Python 3.6
~~~

## Directory description
~~~
Root
├─data/*        where your training data lies
├─saved_model/*      Store the trained models.
├─utils/*     utility functions
├─train_autoencoder.py      train your transformer autoencoder
├─train_clf.py      train your classifier
├─predict_autoencoder.py      test your transformer autoencoder and make some predictions (reconstruct text)
└─predict_FGIM.py      test the fast-gradient-iterative method and control the text generations
~~~

## Data preparation

Put your data in the data directory (note that example data is already given in this repo)
- For training the transformer autoencoder, 2 files are needed: 
	1. train_y, whose format is simply **space-splitted raw text for each line**
	2. vocab, whose format is simple **one vocab for each line**, the 3 special tokens MUST take up the first three lines
- For training the classifier, you need one additional file:
	3. train_y_multilabel, whose format is simple **one label for each line, corresponding to the texts of train_y**

## Model Training

- Train the transformer autoencoder

~~~
python train_autoencoder.py
~~~

- Train the classifier

~~~
python train_clf.py
~~~

## Model Testing

(Note: FGIM stands for "fast gradient iterative method" proposed by the original paper.)
Prepare your data for prediction (you need 2 files: pred_x and pred_label, examples are also given in the root directory)
~~~
python predict_FGIM.py
~~~
If all goes well, you should see some logging output like the following:)
~~~

INFO:tensorflow:Restoring parameters from saved_model/S2S/Trans/model
INFO:tensorflow:Restoring parameters from saved_model/S2S/Trans_clf/model
INFO:tensorflow:Original input text        : ['人生 很 匆忙 ， 有 谁 愿意 停 在 风景 里']
INFO:tensorflow:Original reconstructed text: ['人生 很 匆忙 ， 有 谁 愿意 停 在 风景 里']
INFO:tensorflow:Original --> Target        : ['neutral']-->['negative']
INFO:tensorflow:Original logits: [[4.1589043e-01 9.9899572e-01 2.4580501e-04]]
INFO:tensorflow:epsilon:0.1 =========================
INFO:tensorflow:	iter:1/10, loss:0.551445, logits:[[1.0000000e+00 3.2631732e-32 5.8501858e-12]], output:['我 很 心疼 啊 ， 好 ！ 可惜 谁 从未 离开 了 去 比赛 呢']
INFO:tensorflow:	iter:2/10, loss:0.551445, logits:[[1.0000000e+00 3.2631732e-32 5.8501858e-12]], output:['我 很 心疼 啊 ， 好 ！ 可惜 谁 从未 离开 了 去 比赛 呢']
INFO:tensorflow:	iter:3/10, loss:0.551445, logits:[[1.0000000e+00 3.2631732e-32 5.8501858e-12]], output:['我 很 心疼 啊 ， 好 ！ 可惜 谁 从未 离开 了 去 比赛 呢']
INFO:tensorflow:epsilon:0.25 =========================
INFO:tensorflow:	iter:1/10, loss:0.551445, logits:[[1.0000000e+00 0.0000000e+00 1.2187446e-25]], output:['我 很 可怜 啊 ！ 太 可惜 了 ！ 对不起 家乡 没 我 想着 下雨 了 还 <UNK> 啦']
INFO:tensorflow:	iter:2/10, loss:0.551445, logits:[[1.0000000e+00 0.0000000e+00 1.2187446e-25]], output:['我 很 可怜 啊 ！ 太 可惜 了 ！ 对不起 家乡 没 我 想着 下雨 了 还 <UNK> 啦']
~~~

## Some observations
Personally, I trained my autoencoder and classifier on 4million Weibo corpus (in Chinese) with a 40k vocab, the train_y_label data is generated automatically by another pretrained sentiment classifier. According to some limited hands-on experiences, I found the following results:
- The final results are not satisfying due to problems below: 
	- FGIM does NOT work for every sentence, there are cases where gradient-update simply cannot  take effect
	- The difficulty of classification task also mattters (e.g., I tried on 6-sentiment classification and 3-sentiment classification, the latter generated better results from my observations)
	- Some epsilon works, some not. There is no guarantee that you can always find a proper epsilon (epsilon is the update rate, namely the small factor multiplied by gradients)
	- Fluency is a problem (Applying better decoder probably helps. (e.g., GPT2))
- Of course, the original experiments are held on English data and different classification tasks (e.g., binary classifcation, caption, review rating), which may reasonably lead to different results.
- Any bug detection in this code is welcome.

- Good case (I trained a larger model, 4-layer, 512-dim)
~~~
INFO:tensorflow:Original input text        : ['哥 你 这 寂寞 太 值钱 了']
INFO:tensorflow:Original reconstructed text: ['哥 你 这 寂寞 太 值钱 了']
INFO:tensorflow:Original --> Target        : ['negative']-->['positive']
INFO:tensorflow:Original logits: [[0.90242577 0.8381719  0.00589545]]
INFO:tensorflow:epsilon:1.0 =========================
INFO:tensorflow:	iter:1/10, loss:0.551445, logits:[[0. 0. 1.]], output:['哥 你 这 寂寞 太 唯美 了 ！']
INFO:tensorflow:epsilon:2.0 =========================
INFO:tensorflow:	iter:1/10, loss:0.551445, logits:[[0. 0. 1.]], output:['哥 你 这 创意 超 性感 ！ 太萌 ！']
~~~

