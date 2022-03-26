"""
General utilities to help with implementation
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def reset_seed(number):
    """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
    """
    random.seed(number)
    torch.manual_seed(number)
    return


def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    Inputs:
    - tensor: A torch tensor of shape (3, H, W) with elements in the range [0, 1]

    Returns:
    - ndarr: A uint8 numpy array of shape (H, W, 3)
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    return ndarr


def visualize_dataset(X_data, y_data, samples_per_class, class_list):
    """
    Make a grid-shape image to plot

    Inputs:
    - X_data: set of [batch, 3, width, height] data
    - y_data: paired label of X_data in [batch] shape
    - samples_per_class: number of samples want to present
    - class_list: list of class names (e.g.) ['plane', 'car', 'bird', 'cat',
      'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    Outputs:
    - An grid-image that visualize samples_per_class number of samples per class
    """

    # Protected lazy import.
    from torchvision.utils import make_grid

    img_half_width = X_data.shape[2] // 2
    samples = []
    for y, cls in enumerate(class_list):
        plt.text(
            -4, (img_half_width * 2 + 2) * y + (img_half_width + 2), cls, ha="right"
        )
        idxs = (y_data == y).nonzero().view(-1)
        for i in range(samples_per_class):
            idx = idxs[random.randrange(idxs.shape[0])].item()
            samples.append(X_data[idx])

    img = make_grid(samples, nrow=samples_per_class)
    return tensor_to_image(img)


def detection_visualizer(img, idx_to_class, bbox=None, pred=None):
    """
    Data visualizer on the original image. Support both GT box input and proposal input.
    
    Input:
    - img: PIL Image input
    - idx_to_class: Mapping from the index (0-19) to the class name
    - bbox: GT bbox (in red, optional), a tensor of shape Nx5, where N is
            the number of GT boxes, 5 indicates (x_tl, y_tl, x_br, y_br, class)
    - pred: Predicted bbox (in green, optional), a tensor of shape N'x6, where
            N' is the number of predicted boxes, 6 indicates
            (x_tl, y_tl, x_br, y_br, class, object confidence score)
    """

    # Protected lazy import.
    import cv2

    img_copy = np.array(img).astype('uint8')

    if bbox is not None:
        for bbox_idx in range(bbox.shape[0]):
            one_bbox = bbox[bbox_idx][:4]
            cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2],
                        one_bbox[3]), (255, 0, 0), 2)
            if bbox.shape[1] > 4: # if class info provided
                obj_cls = idx_to_class[bbox[bbox_idx][4].item()]
                cv2.putText(img_copy, '%s' % (obj_cls),
                          (one_bbox[0], one_bbox[1]+15),
                          cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    if pred is not None:
        for bbox_idx in range(pred.shape[0]):
            one_bbox = pred[bbox_idx][:4]
            cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2],
                        one_bbox[3]), (0, 255, 0), 2)
            
            if pred.shape[1] > 4: # if class and conf score info provided
                obj_cls = idx_to_class[pred[bbox_idx][4].item()]
                conf_score = pred[bbox_idx][5].item()
                cv2.putText(img_copy, '%s, %.2f' % (obj_cls, conf_score),
                            (one_bbox[0], one_bbox[1]+15),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    plt.imshow(img_copy)
    plt.axis('off')
    plt.show()


def decode_captions(captions, idx_to_word):
    """
    Decoding caption indexes into words.
    Inputs:
    - captions: Caption indexes in a tensor of shape (Nx)T.
    - idx_to_word: Mapping from the vocab index to word.

    Outputs:
    - decoded: A sentence (or a list of N sentences).
    """
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def attention_visualizer(img, attn_weights, token):
    """
    Visuailze the attended regions on a single frame from a single query word.
    Inputs:
    - img: Image tensor input, of shape (3, H, W)
    - attn_weights: Attention weight tensor, on the final activation map
    - token: The token string you want to display above the image

    Outputs:
    - img_output: Image tensor output, of shape (3, H+25, W)

    """

    # Protected lazy import.
    import cv2

    C, H, W = img.shape
    assert C == 3, "We only support image with three color channels!"

    # Reshape attention map
    attn_weights = cv2.resize(
        attn_weights.data.numpy().copy(),
        (H, W),
        interpolation=cv2.INTER_NEAREST
    )
    attn_weights = np.repeat(np.expand_dims(attn_weights, axis=2), 3, axis=2)

    # Combine image and attention map
    img_copy = (
        img.float().div(255.0).permute(1, 2, 0).numpy()[:, :, ::-1].copy()
    )  # covert to BGR for cv2
    masked_img = cv2.addWeighted(attn_weights, 0.5, img_copy, 0.5, 0)
    img_copy = np.concatenate((np.zeros((25, W, 3)), masked_img), axis=0)

    # Add text
    cv2.putText(
        img_copy,
        "%s" % (token),
        (10, 15),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (255, 255, 255),
        thickness=1,
    )

    return img_copy
