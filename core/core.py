from .ImageSearch.ImageSearch import ImageSearch
from .SensitiveClassify.SensitiveClassify import SensitiveClassify
from .SentimentText.SentimentText import SentimentText

import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO

class Core:
    def __init__(self, db=None):
        self.db = db
        self.search_core = ImageSearch()
        # self.sentiment_core = SensitiveClassify()
        # self.sensitive_core = SentimentText()

        self.init_emmbedding()

    def init_emmbedding(self):
        museums = self.db.museums.find()
        for museum in museums:
            museumId = "M" + museum["museumId"]
            coverUrl = museum["coverUrl"]
            # thumbnailUrl = museum["thumbnailUrl"]
            image = self.prepare_image(coverUrl)
            if not image: continue
            self.search_core.add_image(image, museumId)

        events = self.db.events.find()
        for event in events:
            eventId = "E" + event["eventId"]
            coverUrl = event["coverUrl"]
            # thumbnailUrl = event["thumbnailUrl"]
            image = self.prepare_image(coverUrl)
            if not image: continue
            self.search_core.add_image(image, eventId)

    def prepare_image(self, image: Image.Image):
        if isinstance(image, str):
            if image.startswith("https://"):
                return self.get_image(image)
            return  # only support `https://`
            image = Image.open(image)
        else:
            return image

    def get_image(self, url):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img

    def search_image(self, image, top_k):
        ...

    def search_query(self, query, top_k):
        return self.search_core.search_text(query, top_k)

    def search_similar(self, image, top_k):
        ...

    def sentiment_analysis(self, text):
        ...

    def sensitive_analysis(self, image):
        ...
