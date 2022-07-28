import numpy as np
from numpy.typing import NDArray
from staliro.core.interval import Interval
from staliro.core.model import Model, ModelData, Failure, StaticInput, Signals
from staliro.options import Options, SignalOptions
from staliro.specifications import RTAMTDense
from staliro.staliro import staliro, simulate_model


try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True


AFCDataT = NDArray[np.float_]
AFCResultT = ModelData[AFCDataT, None]


class AFCModel(Model[AFCResultT, None]):
    MODEL_NAME = "AbstractFuelControl_M1"

    def __init__(self) -> None:
        

        if not _has_matlab:
            raise RuntimeError(
                "Simulink support requires the MATLAB Engine for Python to be installed"
            )

        engine = matlab.engine.start_matlab()
        engine.workspace["simTime"] = matlab.double([50])
        model_opts = engine.simget(self.MODEL_NAME)
        engine.workspace['measureTime'] = matlab.double([1.0])
        engine.workspace['fault_time'] = matlab.double([100])
        engine.workspace['eta'] = matlab.double([1.0])
        engine.workspace['h'] = matlab.double([0.05])
        engine.workspace['zeta_min'] = matlab.double([5.0])
        engine.workspace['fuel_inj_tol'] = matlab.double([1.0])
        engine.workspace['MAF_sensor_tol'] = matlab.double([1.0])
        engine.workspace['AF_sensor_tol'] = matlab.double([1.0])
        engine.workspace['C'] = matlab.double([0.05])
        engine.workspace['Cr'] = matlab.double([0.1])
        engine.workspace['Cl'] = matlab.double([0.1])
        engine.workspace['spec_num'] = matlab.double([1])
        engine.workspace['taus'] = matlab.double([11])
        engine.workspace['form_id'] = matlab.double([2])
        engine.workspace['Ut'] = matlab.double([0.008])

        self.sampling_step = 0.05  
                
        self.engine = engine
        self.model_opts = engine.simset(model_opts, "SaveFormat", "Array")

    def simulate(
        self, static: StaticInput, signals: Signals, intrvl: Interval
    ) -> AFCResultT:
        
        
        
        sim_t = matlab.double([0, intrvl.upper])
        n_times = intrvl.length // self.sampling_step + 2
        signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
        w = (signals[0].at_times(signal_times))
        throttle = (signals[1].at_times(signal_times))
        model_input = matlab.double(np.row_stack((signal_times, w, throttle)).T.tolist())
        
        timestamps, _, data = self.engine.sim(
        self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )
        timestamps_array = np.array(timestamps).flatten()
        data_array = np.array(data)
        
        return ModelData(data_array.T, timestamps_array)