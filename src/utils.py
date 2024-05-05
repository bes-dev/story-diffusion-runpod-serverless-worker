import random
import io
import zipfile
import numpy as np
import torch
from PIL import Image


def compress_images_to_zip(images: list[Image.Image]) -> io.BytesIO:
    """ Compress images to a zip file

    Args:
        images (list[Image.Image]): List of images

    Returns:
        io.BytesIO: Zip file
    """
    zip_data = io.BytesIO()
    with zipfile.ZipFile(zip_data, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, image in enumerate(images):
            image_data = io.BytesIO()
            image.save(image_data, format="PNG")
            image_data.seek(0)
            zip_file.writestr(f"{i}.png", image_data.getvalue())
    return zip_data


def setup_seed(seed: int):
    """ Set random seed for reproducibility

    Args:
        seed (int): random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def is_torch2_available() -> bool:
    """ Check if torch2 is available

    Returns:
        bool: True if torch2 is available
    """
    return hasattr(F, "scaled_dot_product_attention")


def cal_attn_mask_xl(
        total_length: int,
        id_length: int,
        sa32: float,
        sa64: float,
        height: int,
        width: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """ Calculate the attention mask for SDXL

    Args:
        total_length (int): Total length
        id_length (int): ID length
        sa32 (float): Attention mask for 32x32
        sa64 (float): Attention mask for 64x64
        height (int): Image height
        width (int): Image width
        device (str): Device (default: "cuda")
        dtype (torch.dtype): Data type (default: torch.float16)

    Returns:
        torch.Tensor: Attention mask
    """
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    bool_matrix1024 = torch.rand((1, total_length * nums_1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((1, total_length * nums_4096),device = device,dtype = dtype) < sa64
    bool_matrix1024 = bool_matrix1024.repeat(total_length,1)
    bool_matrix4096 = bool_matrix4096.repeat(total_length,1)
    for i in range(total_length):
        bool_matrix1024[i:i+1, id_length*nums_1024:] = False
        bool_matrix4096[i:i+1, id_length*nums_4096:] = False
        bool_matrix1024[i:i+1, i*nums_1024:(i+1)*nums_1024] = True
        bool_matrix4096[i:i+1, i*nums_4096:(i+1)*nums_4096] = True
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1,nums_1024,1).reshape(-1,total_length * nums_1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1,nums_4096,1).reshape(-1,total_length * nums_4096)
    return mask1024, mask4096
