def get_model_and_optimizer(cfgs, test=False)
    if cfgs["model"] == "htcdc":
        from htcdc import UBins
        model = UBins(cfgs)
    else:
        NotImplementedError
    if test:
        return model
        
    if cfgs["optimizer"] == "Adam":
        from torch.optim import Adam
        optim = Adam
    elif cfgs["optimizer"] == "SGD":
        from torch.optim import SGD
        optim = SGD
    elif cfgs["optimizer"] == "Nadam":
        from torch.optim import NAdam
        optim = NAdam
    elif cfgs["optimizer"] == "AdamW":
        from torch.optim import AdamW
        optim = AdamW
    else:
        NotImplementedError
    optimizer = optim(filter(lambda x: x.requires_grad, model.parameters()), lr=cfgs["lr"])
    return model, optimizer