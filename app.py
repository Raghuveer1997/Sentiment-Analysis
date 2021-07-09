from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort
import os
import sys


def main():
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'
    from Sentiment_Analysis.model.library.cnn_lstm import WordVecCnnLstm
   

    app = Flask(__name__)
    app.config.from_object(__name__)  # load config from this file , app.py

    # Load default config and override config from an environment variable
    app.config.from_envvar('FLASKR_SETTINGS', silent=True)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    wordvec_cnn_lstm_classifier = WordVecCnnLstm()


    @app.route('/')
    def home():
        return render_template('home.html')

    @app.route('/about')
    def about():
        return 'About Us'


    @app.route('/wordvec_cnn_lstm', methods=['POST', 'GET'])
    def wordvec_cnn_lstm():
        if request.method == 'POST':
            if 'sentence' not in request.form:
                flash('No sentence post')
                redirect(request.url)
            elif request.form['sentence'] == '':
                flash('No sentence')
                redirect(request.url)
            else:
                sent = request.form['sentence']
                sentiments = wordvec_cnn_lstm_classifier.predict(sent)
                value = wordvec_cnn_lstm_classifier.predict_class(sent)
                return render_template('wordvec_cnn_lstm_result.html', sentence=sent, sentiments=sentiments, values = value)
        return render_template('wordvec_cnn_lstm.html')
 

    @app.route('/predict', methods=['POST', 'GET'])
    def measure_sentiment():
        if request.method == 'POST':
            if not request.json or 'sentence' not in request.json or 'network' not in request.json:
                abort(400)
            sentence = request.json['sentence']
            network = request.json['network']
        else:
            sentence = request.args.get('sentence')
            network = request.args.get('network')

        sentiments = []
        if network == 'cnn_lstm':
            sentiments = wordvec_cnn_lstm_classifier.predict(sentence)
        return jsonify({
            'sentence': sentence,
            'pos_sentiment': float(str(sentiments[0])),
            'neg_sentiment': float(str(sentiments[1])),
            'network': network
        })

    @app.errorhandler(404)
    def not_found(error):
        return make_response(jsonify({'error': 'Not found'}), 404)

    model_dir_path = os.path.join(current_dir, './model')

    wordvec_cnn_lstm_classifier.load_model(model_dir_path)

    wordvec_cnn_lstm_classifier.test_run('i liked the Da Vinci Code a lot.')
 
    app.run(debug=True, use_reloader=False)


if __name__ == '__main__':
    main()
