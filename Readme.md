# Dynamic Graph Neural Networks for Student Performance Prediction

### Author: Soheila Farokhi & Arash Azizian Foumani

[//]: # (### Datasets)

[//]: # (All the pre-processing has already been carried out and the result is saved under `data/` folder.)

[//]: # ()
[//]: # (Classification dataset: `classification_mooc.csv`)

[//]: # ()
[//]: # (Regression dataset: `regression_mooc.csv`)

[//]: # ()
[//]: # ()
[//]: # (### Dataset Format)

[//]: # ()
[//]: # (The network has the following format:)

[//]: # (- One line per interaction/edge.)

[//]: # (- Each line includes: *user_id, challenge_id, timestamp, final_score, course_id, exercise_id, difficulty, retry_status, duration*.)

[//]: # (- First line is the network format. )

[//]: # (- *user_id* and *challenge_id* fields are numeric.)

[//]: # (- *Timestamp* is in cardinal format &#40;not in datetime&#41;.)

[//]: # (- *Final_score* in the classification dataset is 0, 1, 2, 3 or 4.)

[//]: # (- *Final_score* in the regression dataset is between 0 and 100.)

[//]: # ()
[//]: # ()
[//]: # (The first few lines of the dataset can be:)

[//]: # (```)

[//]: # (user_id,challenge_id,timestamp,final_score,course_id,exercise_id,retry_status,duration,difficulty)

[//]: # (28049,499,0,0,1205,180,0,0,3)

[//]: # (28049,501,0,0,1205,180,0,0,3)

[//]: # (28049,502,0,0,1205,180,0,0,2    )

[//]: # (28049,504,1,0,1205,180,0,0,2 )

[//]: # (28049,503,1,0,1205,180,0,0,2)

[//]: # (```)

[//]: # (### Code Setup and Requirements)

[//]: # (You can install all the required packages using the following command:)

[//]: # (```)

[//]: # (    $ pip install -r requirements.txt)

[//]: # (```)

[//]: # ()
[//]: # (### Running the JODIE code)

[//]: # ()
[//]: # (To train the JODIE model using the `data/<network>.csv` dataset, use the following command. This will save a model for every epoch in the `saved_models/<network>/` directory.)

[//]: # (```)

[//]: # (   $ python jodie.py --network <network> --classification <classification> --epochs 50)

[//]: # (```)

[//]: # ()
[//]: # (This code can be given the following command-line arguments:)

[//]: # (1. `--network`: this is the name of the file which has the data in the `data/` directory. The file should be named `<network>.csv`. This is a required argument.)

[//]: # (3. `--gpu`: this is the id of the gpu where the model is run. Default value: -1 &#40;to run on the GPU with the most free memory&#41;.)

[//]: # (4. `--epochs`: this is the maximum number of interactions to train the model. Default value: 50.)

[//]: # (5. `--embedding_dim`: this is the number of dimensions of the dynamic embedding. Default value: 128.)

[//]: # (7. `--classification`: this is a boolean input indicating if the training is done on the classification problem or the regression. Default value: True.)

[//]: # ()
[//]: # (### Evaluate JODIE model)

[//]: # ()
[//]: # (#### Grade prediction)

[//]: # ()
[//]: # (To evaluate the performance of the model for the grade prediction classification task, use the following command. The command iteratively evaluates the performance for all epochs of the model and outputs the final test performance. )

[//]: # (```)

[//]: # (    $ chmod +x evaluate_all_epochs.sh)

[//]: # (    $ ./evaluate_all_epochs.sh classification_mooc classification)

[//]: # (```)

[//]: # (This will output the performance numbers to the `results/grade_classification_prediction_classification_mooc.txt` file.)

[//]: # ()
[//]: # (To evaluate the trained model's performance for predicting discrete grade in only one epoch, use the following command.)

[//]: # (```)

[//]: # (   $ python evaluate_classification_grade_prediction.py --network classification_mooc --model jodie --epoch 49)

[//]: # (```)

[//]: # ()
[//]: # (To evaluate the performance of the model for the grade prediction regression task, use the following command. The command iteratively evaluates the performance for all epochs of the model and outputs the final test performance. )

[//]: # (```)

[//]: # (    $ chmod +x evaluate_all_epochs.sh)

[//]: # (    $ ./evaluate_all_epochs.sh regression_mooc regression)

[//]: # (```)

[//]: # (This will output the performance numbers to the `results/grade_regression_prediction_regression_mooc.txt` file.)

[//]: # ()
[//]: # (To evaluate the trained model's performance for predicting continuous grade in only one epoch, use the following command. )

[//]: # (```)

[//]: # (   $ python evaluate_regression_grade_prediction.py --network regression_mooc --model jodie --epoch 49)

[//]: # (```)

[//]: # ()
[//]: # (### References )

[//]: # (*Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks*. Srijan Kumar, Xikun Zhang, Jure Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining &#40;KDD&#41;, 2019. )

[//]: # ()
[//]: # (```)

[//]: # ( @inproceedings{kumar2019predicting,)

[//]: # (	title={Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks},)

[//]: # (	author={Kumar, Srijan and Zhang, Xikun and Leskovec, Jure},)

[//]: # (	booktitle={Proceedings of the 25th ACM SIGKDD international conference on Knowledge discovery and data mining},)

[//]: # (	year={2019},)

[//]: # (	organization={ACM})

[//]: # ( })

[//]: # (```)
