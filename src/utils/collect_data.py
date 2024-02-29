import torch


def collect_data_from_dsets(dsets, num_samples, split = "train"):
    if split == "train":
        data_loader = dsets._train_dataloader()
    elif split == "validation":
        data_loader = dsets._val_dataloader()
    else:
        raise NotImplementedError()

    for batch in data_loader:
        data_shape = batch.size()[1:]
        data_type = batch.type()
        break

    type_mapping = {
        "torch.LongTensor": torch.long
    }

    data = torch.empty([num_samples, *data_shape], dtype = type_mapping[data_type])
    n = 0
    while n < num_samples:
        for batch in data_loader:
            B = min(batch.size(0), num_samples - n)
            data[n:n+B] = batch[0:B]
            n += B

            if n >= num_samples:
                break

    return data
