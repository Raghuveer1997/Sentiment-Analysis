# sentiment-analysis

Web api built on flask for sentiment analysis using Word Embedding, RNN and CNN

# Usage

Run the following command to install the keras, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```

Below shows the sample code in [wordvec_cnn_lstm_train.py]:
```python
import numpy as np
import sys
import os


def main():
    random_state = 42
    np.random.seed(random_state)

    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    output_dir_path = current_dir + '/models'
    data_file_path = current_dir + '/data/umich-sentiment-train.txt'

    from keras_sentiment_analysis.library.cnn_lstm import WordVecCnnLstm
    from keras_sentiment_analysis.library.utility.simple_data_loader import load_text_label_pairs
    from keras_sentiment_analysis.library.utility.text_fit import fit_text

    text_data_model = fit_text(data_file_path)
    text_label_pairs = load_text_label_pairs(data_file_path)

    classifier = WordVecCnnLstm()
    batch_size = 64
    epochs = 20
    history = classifier.fit(text_data_model=text_data_model,
                             model_dir_path=output_dir_path,
                             text_label_pairs=text_label_pairs,
                             batch_size=batch_size, epochs=epochs,
                             test_size=0.3,
                             random_state=random_state)


if __name__ == '__main__':
    main()

```

The above commands will train wordvec_cnn_lstm model on the "data  umich-sentiment-train.txt"



## Running Web Api Server

Goto [templates](templates) directory and run the following command:

```bash
python app.py
```

Now navigate your browser to http://localhost:5000 and you can try out various predictors built with the following
trained classifiers:


* 1-D CNN + LSTM with Word Embedding

## Invoke Web Api

For example, you can get the sentiments for the sentence "i like the Da Vinci Code a lot." by running the following command:

```bash
curl -H 'Content-Type: application/json' -X POST -d '{"network":"lstm_bidirectional_softmax", "sentence":"i like the Da Vinci Code a lot."}' http://localhost:5000/measure_sentiments
```

And the following will be the json response:

```json
{
    "neg_sentiment": 0.0000434154,
    "network": "lstm_bidirectional_softmax",
    "pos_sentiment": 0.999957,
    "sentence": "i like the Da Vinci Code a lot."
}
```









