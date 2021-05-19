import data_loader


if __name__ == "__main__":
    gmd = data_loader.GrooveMidiDataset()
    inputs, outputs, idx = gmd[0]
    print(inputs)
