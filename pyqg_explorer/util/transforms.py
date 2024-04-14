import torch

""" Functions to normalise and denormalise input and output fields """

def normalise_field(field, mean, std):
    """ Map a field (torch tensor) to a normalised space """
    field = field.clone()
    field.sub_(mean).div_(std)
    return field

def denormalise_field(field, mean, std):
    """ Take a normalised field (torch tensor), denormalise it """
    field = field.clone()
    field.mul_(std).add_(mean)
    return field

def normalise(self,q,config):
    ## Map from physical to normalised space using the factors used to train the network
    ## Normalise each field individually, then cat arrays back to shape appropriate for a torch model
    x_upper = normalise_field(q[0],config["q_mean_upper"],network.config["q_std_upper"])
    x_lower = normalise_field(q[1],config["q_mean_lower"],network.config["q_std_lower"])
    x = torch.stack((x_upper,x_lower),dim=0)
    return x

def denormalise(self,q,config):
    ## Map from physical to normalised space using the factors used to train the network
    ## Normalise each field individually, then cat arrays back to shape appropriate for a torch model
    x_upper = denormalise_field(q[0],network.config["q_mean_upper"],network.config["q_std_upper"])
    x_lower = denormalise_field(q[1],network.config["q_mean_lower"],network.config["q_std_lower"])
    x = torch.stack((x_upper,x_lower),dim=0)
    return x


def produce_standard_q(model,x_data):
    """ For a network trained on residuals, perform the necessary renormalisations to produce a 
        standardised potential vorticity field """
    
    residuals=model(x_data)
    up=residuals[:,0,:,:]
    low=residuals[:,1,:,:]

    ## Transform from residual space to physical space
    up_phys=denormalise_field(up,model.config["res_mean_upper"],model.config["res_std_upper"])+denormalise_field(x_data[:,0,:,:],model.config["q_mean_upper"],model.config["q_std_upper"])
    up_norm=normalise_field(up_phys,model.config["q_mean_upper"],model.config["q_std_upper"])
    ## Transform from residual space to physical space
    low_phys=denormalise_field(low,model.config["res_mean_lower"],model.config["res_std_lower"])+denormalise_field(x_data[:,1,:,:],model.config["q_mean_lower"],model.config["q_std_lower"])
    low_norm=normalise_field(low_phys,model.config["q_mean_lower"],model.config["q_std_lower"])

    return residuals,torch.cat((up_norm.unsqueeze(1),low_norm.unsqueeze(1)),1)

def map_residual_to_q(model,field,x_data):
    up=field[:,0,:,:]
    low=field[:,1,:,:]

    ## Transform from residual space to physical space
    up_phys=denormalise_field(up,model.config["res_mean_upper"],model.config["res_std_upper"])+denormalise_field(x_data[:,0,:,:],model.config["q_mean_upper"],model.config["q_std_upper"])
    up_norm=normalise_field(up_phys,model.config["q_mean_upper"],model.config["q_std_upper"])
    ## Transform from residual space to physical space
    low_phys=denormalise_field(low,model.config["res_mean_lower"],model.config["res_std_lower"])+denormalise_field(x_data[:,1,:,:],model.config["q_mean_lower"],model.config["q_std_lower"])
    low_norm=normalise_field(low_phys,model.config["q_mean_lower"],model.config["q_std_lower"])

    return torch.cat((up_norm.unsqueeze(1),low_norm.unsqueeze(1)),1)

    