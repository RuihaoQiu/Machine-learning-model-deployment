from model.preprocess import origin_tokenize, make_poss, sent2features
from model.postprocess import label_original_tokens, export_html

import joblib

model_path = "model/crf-50.joblib"


def get_model_api():
    """Returns lambda function for api"""

    # 1. initialize model once and for all
    crf = joblib.load(model_path)

    def model_api(input_data):
        """
        Args:
            input_data: submitted to the API, raw string

        Returns:
            output_data: after some transformation, to be
                returned to the API

        """
        texts = [input_data]
        original_tokenss = [origin_tokenize(x) for x in texts]
        data_poss = make_poss(original_tokenss)
        X_in = [sent2features(s) for s in data_poss]
        y_pred = crf.predict(X_in)
        labeled_tokenss = label_original_tokens(original_tokenss, y_pred)
        output_data = export_html(labeled_tokenss[0])

        return output_data

    return model_api
