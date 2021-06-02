def get_class(loss_class):
    if loss_class == "pairwise":
        return PairWise(margin=1)
    elif loss_class == "pointwise":
        return PointWise()
    raise ValueError('Invalid loss class: %s' % loss_class)
        
        
from .pairwise import PairWise
from .pointwise import PointWise