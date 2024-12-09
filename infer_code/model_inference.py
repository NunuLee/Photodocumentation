import numpy as np

def apt_inference(dataloader, model):

    for input in dataloader:
        input = input.numpy()
        output = model(input, 1)
        output = output[0]
        output = np.reshape(output, (18,))
        top_class = np.where(output==max(output))[0][0]

    return top_class