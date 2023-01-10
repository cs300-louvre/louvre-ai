from nudenet import NudeClassifier

class SensitiveClassify:
    def __init__(self):
        self.classifier = NudeClassifier()

    def analysis(self, image_path):
        return self.classifier.classify(image_path)
