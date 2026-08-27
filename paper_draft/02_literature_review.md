# 2. Literature Review

## 2.1 Ambient-sensor activity recognition

Activity recognition in smart homes has been studied with probabilistic models, classical machine learning, and deep neural networks. Early systems often used handcrafted rules or Hidden Markov Models. These methods offered clear temporal structure and were easier to interpret. Their performance depended on prior knowledge and carefully defined probability relationships.

A recent unsupervised study combined two Hidden Markov Models. One model estimated the room occupied by each resident. The second model recognised the activity. The method was tested on the multi-resident SDHAR-HOME dataset and achieved accuracies between 86.78% and 91.68%. Its main strength was reduced dependence on activity labels. Its main limitation was the need for additional resident-location information in the multi-user setting.

Classical machine learning remains competitive on Aruba. Fixed-time windows have been used to extract spatiotemporal features such as time of day, sensor activation counts, and the number of distinct active sensors. Random Forest, XGBoost, and SVM reached high accuracy. The best result was 97% in an easier setting. A more realistic experiment retained the `Other` class and reduced the sensor set. Accuracy fell to 89%. This difference shows that raw continuous streams are harder than pre-segmented activity instances.

## 2.2 Deep sequence models

Recurrent models have been widely used because activity data are sequential. LSTM and BiLSTM networks can model longer dependencies than basic recurrent networks. Hybrid CNN-recurrent models add local feature extraction before temporal modelling. Multi-branch CNN-BiLSTM-BiGRU models have reported high accuracy on wearable datasets. These results show the value of local and long-range temporal features. However, wearable signals are regularly sampled. Ambient smart-home data are event-triggered and irregular.

Transformers offer a different approach. Self-attention can relate distant events without recurrent processing. A GPT-style decoder has been trained on sensor activation sequences from Aruba, Milan, and Cairo. The pretraining task predicts the next sensor activation. The learned embedding improved balanced accuracy and showed transfer between homes. The result confirms that sensor events can be treated as a symbolic language. It also shows that self-supervised learning is useful when labelled smart-home data are limited.

## 2.3 Preprocessing, imbalance, and augmentation

Performance is strongly affected by preprocessing. Convolutional autoencoders have been combined with normalisation, magnitude transformation, PCA, and resampling techniques. These methods improved results on smartphone HAR data. The lesson for ambient sensing is not to copy the same transformations. The important point is that preprocessing and class balance must be treated as part of the model design.

Generative augmentation has also improved sensor-based HAR. Conditional Wasserstein GANs have been used to create synthetic accelerometry sequences. The largest gains appeared when the real dataset was small. Ambient sensor events are symbolic rather than continuous. Direct use of an accelerometer GAN is therefore not appropriate. Event-sequence generation requires a different model that preserves sensor order, time gaps, and physical plausibility.

## 2.4 Research gap

The reviewed work leaves four connected gaps.

First, most Aruba studies rely on one fixed context length. A single window cannot suit both short transition activities and long routines.

Second, irregular time gaps are often appended as an ordinary feature. They are rarely used to control the memory dynamics of the sequence model.

Third, sensor identity improves within-home accuracy but reduces transferability. Sensor type and room semantics can provide a more stable representation across homes.

Fourth, random splits of heavily overlapping windows can produce optimistic results. A chronological split must be completed before window generation.

The proposed work addresses these gaps through adaptive multi-scale context, continuous-time state retention, semantic event factorisation, self-supervised auxiliary learning, and a chronological evaluation protocol.
