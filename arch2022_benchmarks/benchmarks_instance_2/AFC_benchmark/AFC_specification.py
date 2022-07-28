import numpy as np
from staliro.options import SignalOptions
from staliro.specifications import RTAMTDense
from staliro.signals import piecewise_constant

def load_specification_dict(benchmark):

    
    

    rise = "(theta <= 8.8) and (F[0,0.05] (theta >= 40))"
    fall = "(theta >= 40) and (F[0,0.05] (theta <= 8.8))"
    mod_u_1 = "G[1,5] ((ut <= 0.008) and (ut >= -0.008))"
    AFC27_phi = f"G[11,50] (({rise} or {fall}) -> ({mod_u_1}))"
    


    mod_u_2 = "(ut <= 0.007) and (ut >= -0.007)"
    AFC29_phi = f"G[11,50] ({mod_u_2})"
    
    


    mod_u_3 = "(ut <= 0.007) and (ut >= -0.007)"
    AFC33_phi = f"G[11,50] ({mod_u_3})"



    spec_dict = {
        "AFC27": RTAMTDense(AFC27_phi, {"theta": 2, "ut": 0}),
        "AFC29": RTAMTDense(AFC29_phi, {"ut": 0}),
        "AFC33": RTAMTDense(AFC33_phi, {"ut": 0}),
        }


    if benchmark not in spec_dict.keys():
        raise ValueError(f"Inappropriate Benchmark name :{benchmark}. Expected one of {spec_dict.keys()}")
    
    if benchmark == "AFC27" or benchmark == "AFC29":
        signals = [
        SignalOptions(control_points=[(900, 1100)], factory=piecewise_constant),
        SignalOptions(control_points= [(0, 61.2)] * 10, signal_times=np.linspace(0.,50.,10), factory=piecewise_constant),
        ]
    elif benchmark == "AFC33":
        signals = [
        SignalOptions(control_points=[(900, 1100)], factory=piecewise_constant),
        SignalOptions(control_points= [(61.2, 81.2)] * 10, signal_times=np.linspace(0.,50.,10, endpoint=False), factory=piecewise_constant),
        ]

    return spec_dict[benchmark], signals