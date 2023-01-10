from typing import Union
import os
import re
import torch
import clip
from PIL import Image
from tqdm import tqdm
import pandas as pd


def normalize_text(text):
    ''' Normalize text '''
    text = text.lower()
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = text.strip()
    return text


class ImageSearch:
    def __init__(self,
                 model_path: str = "ViT-B/32",
                 image_dir: str = "images",
                 embed_dir: str = "embedding",
                 temp_dir='temp',
                 device: str = 'cuda'
                 ):
        '''
        Args:
            model_path: path to model
            image_dir: path to image directory
            embed_dir: path to embedding directory
            temp_dir: path to temp directory
            device: device to use
        '''

        self.device = "cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu"
        self.image_dir = image_dir
        self.embed_dir = embed_dir
        self.model, self.preprocess = clip.load(model_path, device=device)
        self.model.eval()
        self.model.to(device)
        self.temp_dir = temp_dir
        self.is_change = True

        os.makedirs(self.temp_dir, exist_ok=True)
        if os.path.exists(os.path.join(temp_dir, "embedding.csv")):
            self.emebed_file = pd.read_csv(os.path.join(temp_dir, "embedding.csv"))
        else:
            self.emebed_file = pd.DataFrame(columns=["image_name"])

    def add_image(self, image: Image.Image, image_name: str):
        '''
        Add image to image directory
        Args:
            image: image to add
            image_name: name of image
        '''
        if image_name in self.emebed_file["image_name"].values:
            return
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        torch.save(image_features, os.path.join(self.temp_dir, image_name + ".pt"))
        self.emebed_file.loc[len(self.emebed_file)] = [image_name]
        self.emebed_file.to_csv(os.path.join(self.temp_dir, "embedding.csv"), index=False)
        self.is_change = True

    def load_features(self):
        if not self.is_change:
            return
        self.image_features = []
        for image_name in tqdm(self.emebed_file["image_name"].values):
            image_features = torch.load(os.path.join(self.temp_dir, image_name + ".pt"))
            self.image_features.append(image_features)
        self.image_features = torch.cat(self.image_features, dim=0)
        self.is_change = False

    def search_text(self, text, top_k):
        self.load_features()
        text = normalize_text(text)
        text = clip.tokenize([text]).to(self.device)
        query_features = self.model.encode_text(text)
        image_features = self.image_features
        similarity = (100.0 * image_features @ query_features.T)
        top_k = min(len(self.emebed_file["image_name"].values), top_k)
        values, indices = similarity.topk(top_k, dim=0)
        # load image name
        result = []
        for value, index in zip(values, indices):
            result.append(self.emebed_file["image_name"].values[index])
        return result
