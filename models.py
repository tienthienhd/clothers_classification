import pickle
import numpy as np
import preprocessing
import requests
from cnn import CNN

mapping = {
    0: 'camo',
    1: 'color_block',
    2: 'baroque',
    3: 'solid_color',
    4: 'leopard'
}


class Models(object):
    def __init__(self):
        self.preprocessing_tranferlearning = preprocessing.PretrainedModel()
        self.svm = pickle.load(open('model_saved/svm.model', 'rb'))
        self.knn = pickle.load(open('model_saved/knn.model', 'rb'))
        self.cnn = CNN()
        self.current_image_url = None

    def set_image_url(self, local_url):
        self.current_image_url = local_url

    def download(self, url_image):
        '''
        :param url_image: if image not on local them download
        :return: url local of image
        '''
        idx = url_image.rfind('/')
        url_local = "static/image_test/" + url_image[idx+1:]
        if url_image.startswith('http'):
            print('Save file download to', url_local)
            # download
            with open(url_local, 'wb') as f:
                respone = requests.get(url_image, stream=True, timeout=10)
                if not respone.ok:
                    raise Exception('Cannot connect to download image.')
                print(respone)
                for block in respone.iter_content(1024):
                    if not block:
                        break
                    f.write(block)

            print('Downloaded '+ url_image)
            self.current_image_url = url_local
            return url_local
        self.current_image_url = url_image
        return url_image

    def predict(self):
        # tranfer learning
        feature = self.preprocessing_tranferlearning.get_extracted_feature(
            self.current_image_url)
        feature = np.reshape(feature, (1, len(feature)))
        result_svm = mapping[self.svm.predict(feature)[0]]
        result_knn = mapping[self.knn.predict(feature)[0]]

        # cnn
        result_cnn = self.cnn.predict(self.current_image_url)
        print(result_cnn)
        self.current_image_url = None
        return result_svm, result_knn, result_cnn

# model = Models()
# model.predict('data/raw_data/color_block/Anorak000006.jpg')