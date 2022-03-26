import numpy as np
import torch
import torch.nn as nn
import torchvision
import cv2
import matplotlib.pyplot as plt
import random

colormap = []
colormap.append(([105, 0, 31], -1.0))
colormap.append(([113, 2, 32], -0.984))
colormap.append(([118, 4, 33], -0.968))
colormap.append(([123, 6, 34], -0.952))
colormap.append(([131, 9, 35], -0.936))
colormap.append(([135, 10, 35], -0.92))
colormap.append(([141, 12, 37], -0.904))
colormap.append(([147, 14, 38], -0.888))
colormap.append(([154, 16, 39], -0.872))
colormap.append(([159, 17, 39], -0.856))
colormap.append(([164, 19, 40], -0.84))
colormap.append(([172, 21, 41], -0.824))
colormap.append(([177, 24, 42], -0.808))
colormap.append(([181, 31, 46], -0.792))
colormap.append(([183, 34, 48], -0.776))
colormap.append(([186, 40, 50], -0.76))
colormap.append(([188, 45, 52], -0.744))
colormap.append(([192, 52, 56], -0.728))
colormap.append(([194, 57, 58], -0.712))
colormap.append(([197, 63, 60], -0.696))
colormap.append(([199, 67, 63], -0.68))
colormap.append(([203, 74, 66], -0.664))
colormap.append(([205, 79, 68], -0.648))
colormap.append(([209, 86, 73], -0.632))
colormap.append(([212, 92, 74], -0.616))
colormap.append(([214, 96, 77], -0.6))
colormap.append(([217, 104, 83], -0.584))
colormap.append(([218, 107, 85], -0.568))
colormap.append(([221, 113, 89], -0.552))
colormap.append(([223, 117, 93], -0.536))
colormap.append(([226, 124, 98], -0.52))
colormap.append(([228, 129, 102], -0.504))
colormap.append(([230, 133, 106], -0.488))
colormap.append(([232, 139, 110], -0.472))
colormap.append(([235, 145, 115], -0.456))
colormap.append(([237, 150, 118], -0.44))
colormap.append(([240, 157, 123], -0.424))
colormap.append(([242, 161, 127], -0.408))
colormap.append(([244, 166, 132], -0.392))
colormap.append(([245, 172, 138], -0.376))
colormap.append(([245, 175, 143], -0.36))
colormap.append(([246, 178, 148], -0.344))
colormap.append(([247, 182, 152], -0.328))
colormap.append(([248, 188, 160], -0.312))
colormap.append(([248, 192, 164], -0.296))
colormap.append(([249, 195, 169], -0.28))
colormap.append(([250, 201, 176], -0.264))
colormap.append(([250, 205, 181], -0.248))
colormap.append(([251, 210, 188], -0.232))
colormap.append(([252, 213, 192], -0.216))
colormap.append(([252, 217, 197], -0.2))
colormap.append(([252, 220, 200], -0.184))
colormap.append(([252, 223, 205], -0.168))
colormap.append(([251, 224, 209], -0.152))
colormap.append(([251, 226, 213], -0.136))
colormap.append(([250, 228, 215], -0.12))
colormap.append(([250, 232, 220], -0.104))
colormap.append(([249, 233, 224], -0.088))
colormap.append(([249, 236, 229], -0.072))
colormap.append(([248, 237, 231], -0.056))
colormap.append(([248, 239, 235], -0.04))
colormap.append(([247, 243, 240], -0.024))
colormap.append(([247, 244, 243], 0.0))
colormap.append(([246, 246, 246], 0.0))
colormap.append(([244, 245, 246], 0.024))
colormap.append(([241, 244, 246], 0.04))
colormap.append(([237, 242, 245], 0.056))
colormap.append(([234, 241, 244], 0.072))
colormap.append(([232, 240, 244], 0.088))
colormap.append(([228, 238, 243], 0.104))
colormap.append(([226, 237, 243], 0.12))
colormap.append(([222, 235, 242], 0.136))
colormap.append(([220, 234, 241], 0.152))
colormap.append(([217, 233, 241], 0.168))
colormap.append(([214, 231, 241], 0.184))
colormap.append(([210, 229, 240], 0.2))
colormap.append(([208, 228, 240], 0.216))
colormap.append(([204, 226, 238], 0.232))
colormap.append(([199, 223, 237], 0.248))
colormap.append(([193, 221, 235], 0.264))
colormap.append(([189, 218, 234], 0.28))
colormap.append(([182, 215, 232], 0.296))
colormap.append(([177, 212, 230], 0.312))
colormap.append(([173, 211, 230], 0.328))
colormap.append(([168, 207, 228], 0.344))
colormap.append(([163, 205, 226], 0.36))
colormap.append(([158, 202, 225], 0.376))
colormap.append(([154, 201, 224], 0.392))
colormap.append(([148, 197, 222], 0.408))
colormap.append(([143, 195, 221], 0.424))
colormap.append(([138, 192, 219], 0.44))
colormap.append(([129, 187, 216], 0.456))
colormap.append(([124, 183, 214], 0.472))
colormap.append(([116, 178, 211], 0.488))
colormap.append(([112, 175, 209], 0.504))
colormap.append(([106, 171, 208], 0.52))
colormap.append(([101, 168, 206], 0.536))
colormap.append(([92, 163, 203], 0.552))
colormap.append(([86, 159, 201], 0.568))
colormap.append(([81, 155, 200], 0.584))
colormap.append(([76, 152, 198], 0.6))
colormap.append(([68, 147, 195], 0.616))
colormap.append(([65, 144, 194], 0.632))
colormap.append(([61, 139, 191], 0.648))
colormap.append(([59, 137, 189], 0.664))
colormap.append(([57, 133, 188], 0.68))
colormap.append(([53, 129, 185], 0.696))
colormap.append(([52, 126, 184], 0.712))
colormap.append(([49, 123, 182], 0.728))
colormap.append(([47, 120, 181], 0.744))
colormap.append(([43, 116, 178], 0.76))
colormap.append(([41, 112, 177], 0.776))
colormap.append(([39, 109, 176], 0.792))
colormap.append(([36, 106, 174], 0.808))
colormap.append(([32, 101, 171], 0.824))
colormap.append(([31, 97, 166], 0.84))
colormap.append(([28, 93, 159], 0.856))
colormap.append(([25, 89, 153], 0.872))
colormap.append(([24, 85, 148], 0.888))
colormap.append(([21, 79, 141], 0.904))
colormap.append(([19, 76, 136], 0.92))
colormap.append(([17, 72, 130], 0.936))
colormap.append(([15, 69, 126], 0.952))
colormap.append(([12, 63, 118], 0.968))
colormap.append(([10, 59, 113], 0.984))
colormap.append(([9, 56, 108], 1.0))


def colormap_to_weight(colormap, color):
    closest_weight = -1
    closest_dist = 99999999

    for colormap_value, weight in colormap:
        dists = (
            (colormap_value[0] - color[0]) ** 2
            + (colormap_value[1] - color[1]) ** 2
            + (colormap_value[2] - color[2]) ** 2
        )

        if dists <= closest_dist:

            closest_dist = dists
            closest_weight = weight

    return closest_weight


def get_w1(img_path):
    templates = []
    neural_net = cv2.imread(img_path)
    neural_net = cv2.cvtColor(neural_net, cv2.COLOR_BGR2RGB)
    neural_net = cv2.resize(
        neural_net, (900, 800), interpolation=cv2.INTER_AREA
    )
    X_tl = [150] * 7
    Y_tl = [150 + 75 * i for i in range(7)]
    for x, y in zip(X_tl, Y_tl):
        t = neural_net[y + 4 : y + 60, x + 4 : x + 60, :]
        t = cv2.resize(t, (28, 28), interpolation=cv2.INTER_AREA)

        t_one_channel = np.zeros((t.shape[0], t.shape[0]))
        for w in range(t.shape[0]):
            for h in range(t.shape[1]):
                colormap_to_weight(colormap, t[w, h, :])
                t_one_channel[w, h] = colormap_to_weight(colormap, t[w, h, :])
        templates.append(t_one_channel)
    return templates


def get_w2(img_path):
    neural_net = cv2.imread(img_path)
    neural_net = cv2.cvtColor(neural_net, cv2.COLOR_BGR2RGB)
    neural_net = cv2.resize(
        neural_net, (900, 800), interpolation=cv2.INTER_AREA
    )
    w2 = neural_net[165 : 10 * 32 + 165 : 32, 405 : 7 * 32 + 405 : 32, :]
    w2_one_channel = np.zeros((w2.shape[0], w2.shape[1]))
    for w in range(w2.shape[0]):
        for h in range(w2.shape[1]):
            w2_one_channel[w, h] = colormap_to_weight(colormap, w2[w, h, :])
    return w2_one_channel


def display_templates(templates):
    for num, template in enumerate(templates):
        plt.subplot(1, len(templates), num + 1)
        plt.title(str(num))
        plt.axis("off")
        plt.imshow(template, cmap="gray")


def display_w2(w2):
    plt.imshow(w2, cmap="gray")


def colormap_to_weights(x):
    x_one_channel = np.zeros((x.shape[0], x.shape[1]))
    for w in range(x.shape[0]):
        for h in range(x.shape[1]):

            if x[w, h, 0] < x[w, h, 2]:
                x_one_channel[w, h] = np.absolute(x[w, h, 1] - 246) + 246
            else:
                x_one_channel[w, h] = x[w, h, 1]

    return x_one_channel


def evaluate_MNIST(w1, w2):
    w_1 = [torch.from_numpy(t).float() for t in w1]
    w_2 = torch.from_numpy(w2).float()

    loss_function = nn.CrossEntropyLoss()
    nonlinearity = nn.ReLU()
    softmax = nn.Softmax(dim=0)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    MNIST_test_set = torchvision.datasets.MNIST(
        "mnist_dataset/", train=False, download=True, transform=transform
    )

    test_loader = torch.utils.data.DataLoader(MNIST_test_set, 64, shuffle=True)

    correct_digits = []
    incorrect_digits = []

    num_correct = 0
    num_total = 0

    for x, y in test_loader:
        for i in range(x.shape[0]):
            digit, label = x[i, 0, :, :], y[i]

            with torch.no_grad():
                h1 = [torch.sum(digit * t) for t in w_1]
                h1 = nonlinearity(torch.tensor(h1).float())
                out = softmax(torch.matmul(h1, torch.t(w_2)))
                loss = loss_function(out.unsqueeze(0), label.unsqueeze(0))

            if torch.argmax(out).item() == label.item():
                correct_digits.append([digit.numpy(), loss.item()])
                num_correct += 1
            else:
                incorrect_digits.append([digit.numpy(), loss.item()])

            num_total += 1
            accuracy = num_correct / num_total

    correct_digits.sort(key=lambda x: x[1])
    incorrect_digits.sort(key=lambda x: x[1], reverse=True)

    correct_digits = [x[0] for x in correct_digits]
    incorrect_digits = [x[0] for x in incorrect_digits]

    return accuracy, correct_digits, incorrect_digits


def visualize_MNIST():

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    MNIST_train_set = torchvision.datasets.MNIST(
        "mnist_dataset/", train=True, download=True, transform=transform
    )

    num_row = 5
    num_col = 7
    fig, axes = plt.subplots(
        num_row, num_col, figsize=(1.5 * num_col, 2 * num_row)
    )

    for i in range(num_row * num_col):
        random_index = random.randint(0, 10000)
        ax = axes[i // num_col, i % num_col]
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.imshow(
            MNIST_train_set[random_index][0].numpy()[0, :, :], cmap="gray"
        )
        ax.set_title(
            "Label: {}".format(MNIST_train_set[random_index][1])
        )

    plt.tight_layout()
    plt.show()

    return
