def freeze(network):
    for p in network.parameters():
        p.requires_grad = False
    return network

def unfreeze(network):
    for p in network.parameters():
        p.requires_grad = True
    return network