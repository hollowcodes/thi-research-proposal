
import torch
import torchvision
import numpy as np
import imgaug.augmenters as iaa
import random
import matplotlib.pyplot as plt

torch.manual_seed(0)


class CifarTenDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str="", class_amount: int=10):
        """ constructor
        :param str dataset_path: dataset path
        :param int class_amount: amount of classes in the dataset
        """

        self.dataset_path = dataset_path
        self.class_amount = class_amount
        self.dataset = np.load(self.dataset_path, allow_pickle=True)

    def _flip(self, image, chance: float=0.5):
        flip = iaa.Fliplr(chance)
        
        return flip(image=image)
        
    def _create_one_hot(self, int_representation: int) -> list:
        """ create one-hot encoding of the target
        :param int int_representation: class of sample
        :return list: ie. int_representation = 2 -> [0, 0, 1, ..., 0]
        """

        one_hot_target = np.zeros((self.class_amount))
        one_hot_target[int_representation] = 1

        return one_hot_target

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ get sample (batch) from dataset
        :param int idx: index of dataset (iterator of training-loop)
        :return tensor: preprocessed sample and target
        """

        # not using the target
        image, target = self.dataset[idx][0], self.dataset[idx][1]

        # bring image to range [0; 1] and normalize
        image = image / 255
        normalization = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
        image = normalization(torch.Tensor(image).transpose(2, 0))

        # create index longtensor target for CrossEntrpopyLoss
        target = torch.Tensor([target]).type(torch.LongTensor)

        return image, target

    def __len__(self):
        """ returns length of dataset """
        
        return len(self.dataset)



def create_dataloader(dataset_path: str="", batch_size: int=32, class_amount: int=10):
    """ create three dataloader (train, validation, test)
    :param str dataset_path: path to dataset
    :param int batch_size: batch-size
    :return torch.Dataloader: train-, val- and test-dataloader
    """

    dataset = CifarTenDataset(dataset_path=dataset_path, class_amount=class_amount)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=0,
        shuffle=True
    )

    return dataloader



def visualize_augmentation(dataset_path: str=""):
    dataset = np.load(dataset_path, allow_pickle=True)

    rand_idx = random.randint(0, len(dataset) - 1)
    print(rand_idx)
    image = dataset[39258][0]

    plt.rcParams["figure.figsize"] = 30, 30


    augmented_images = []
    augmented_images.extend(iaa.Sequential([iaa.Fliplr(1)])(images=[image]))
    augmented_images.extend(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.075*255), per_channel=0.5)(images=[image]))
    #augmented_images.extend(iaa.GaussianBlur(sigma=(0, 0.75))(images=[image]))
    augmented_images.extend(iaa.Crop(percent=(0, 0.25))(images=[image]))

    #images = seq(images=[image])
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(augmented_images[0])
    axs[1].imshow(augmented_images[1])
    axs[2].imshow(augmented_images[2])
    #axs[3].imshow(augmented_images[3])

    #plt.imshow(augmented_images[0])
    [axs[i].axis("off") for i in range(3)]
    plt.show()


visualize_augmentation("dataset/preprocessed-dataset/train_dataset.npy")




