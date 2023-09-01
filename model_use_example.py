from models.pretrained_networks import EfficientNet_B0_320
from utils.training_utils import apply_net
from utils.data_processing import CodesProcessor
from torch.utils.data import DataLoader
from utils.image_dataset_reader import HistImagesDataset
from torchvision import transforms

path_to_tissues = (
        {'folder': "path_to_images", 'label': 'liver', 'ext': 'png'},
     )

cnn_model_path = 'path_to_trained_model' # pretrained BIHN model, *.pt file

n_samples_per_folder = 3 # number of image samples to test on
n_classes = 16 # number of classes the model was trained on
dev = "cpu"

tr_normalize = transforms.Normalize(mean=(0.5788, 0.3551, 0.5655), std=(1, 1, 1))
transforms_seq = transforms.Compose([transforms.ToTensor(), tr_normalize])
images_dataset = HistImagesDataset(*path_to_tissues, n_samples=n_samples_per_folder, transform=transforms_seq)
test_data_loader = DataLoader(images_dataset)

model = EfficientNet_B0_320(path_trained_model=cnn_model_path, n_classes=n_classes, dev=dev).to(dev)
code_processor = CodesProcessor()
apply_net(model, dev, test_data_loader, verbose=True, code_processor=code_processor)
features = code_processor.get_codes()

print(f"There are {features.shape[0]} feature vectors of length {features.shape[1]}")




