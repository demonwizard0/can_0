import torch
from PIL import Image
from torchvision import transforms

from ben2 import BEN_Base

from modules.background_removal.rmbg_manager import BackgroundRemovalService

class BEN2BackgroundRemovalService(BackgroundRemovalService):
    def _initialize_model_and_transforms(self) -> tuple[BEN_Base, transforms.Compose]:
        """
        Initialize BEN2 model and transforms.
        """
        model: BEN_Base | None = None

        transform = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size), 
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

        return model, transform
    
    def _load_model(self) -> BEN_Base:
        """
        Load the BEN2 background removal model.
        """
        revision = self.model_versions.get_revision(self.settings.model_id)
        model = BEN_Base.from_pretrained(self.settings.model_id, revision=revision)
        return model.to(self.device).eval()

    def _remove_background(self, image: Image) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Remove the background from the image.
        """
        rgb_image = image.convert('RGB').resize(self.settings.input_image_size)

        with torch.no_grad():
            foreground = self.model.inference(rgb_image.copy())
        foreground_tensor = self.transforms(foreground)
        tensor_rgb = foreground_tensor[:3]
        mask = foreground_tensor[-1]

        return tensor_rgb, mask