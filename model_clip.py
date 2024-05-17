import numpy as np
import open_clip
import torch


class Model:
    def __init__(self, display_size=96):
        self.display_size = display_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32',
                                                                               pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        # this should return True if CUDA is available
        print(torch.cuda.is_available())
        # loading features
        features = np.load("E:\\Stuff\\features.npy")
        # Convert the numpy array to a PyTorch tensor and ensure the tensor is of type float for any further computation
        self.features = torch.from_numpy(features).float()

    '''
    The returned indices are indices of the corresponding images in sorted(os.listdir('data'))
    '''

    def search_clip(self, text: str) -> list[int]:
        query = self.tokenizer(text)

        with torch.no_grad():
            text_features = self.model.encode_text(query).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = text_features @ self.features.T
            sorted_indices = sorted(range(len(similarities[0])), key=lambda i: similarities[0][i].item())[::-1]

        return sorted_indices
