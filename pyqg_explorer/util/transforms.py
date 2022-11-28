""" Functions to normalise and denormalise input and output fields """

def normalise_field(field, mean, std):
    """ Map a field in the form of a torch tensor to a normalised space """
    field = field.clone()
    field.sub_(mean).div_(std)
    return field

def denormalise_field(field, mean, std):
    """ Take a normalised field (torch tensor), denormalise it """
    field = field.clone()
    field.mul_(std).add_(mean)
    return field
