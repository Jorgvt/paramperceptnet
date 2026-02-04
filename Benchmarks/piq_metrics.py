from piq import SSIMLoss
from piq import MultiScaleSSIMLoss
from piq import InformationWeightedSSIMLoss
from piq import VIFLoss
from piq import FSIMLoss
from piq import SRSIMLoss
from piq import GMSDLoss
from piq import MultiScaleGMSDLoss
from piq import VSILoss
from piq import DSSLoss
from piq import HaarPSILoss
from piq import MDSILoss
from piq import LPIPS
from piq import PieAPP
from piq import DISTS
from piq import StyleLoss
from piq import ContentLoss

def get_all_piq_full_reference_metrics(**kwargs):
    return {
        "SSIMLoss": SSIMLoss(**kwargs),
        "MultiScaleSSIMLoss": MultiScaleSSIMLoss(**kwargs),
        "InformationWeightedSSIMLoss": InformationWeightedSSIMLoss(**kwargs),
        "VIFLoss": VIFLoss(**kwargs),
        "FSIMLoss": FSIMLoss(**kwargs),
        "SRSIMLoss": SRSIMLoss(**kwargs),
        "GMSDLoss": GMSDLoss(**kwargs),
        "MultiScaleGMSDLoss": MultiScaleGMSDLoss(**kwargs),
        "VSILoss": VSILoss(**kwargs),
        "DSSLoss": DSSLoss(**kwargs),
        "HaarPSILoss": HaarPSILoss(**kwargs),
        "MDSILoss": MDSILoss(**kwargs),
        "LPIPS": LPIPS(**kwargs),
        "PieAPP": PieAPP(**kwargs),
        "DISTS": DISTS(**kwargs),
        "StyleLoss": StyleLoss(**kwargs),
        "ContentLoss": ContentLoss(**kwargs),
    }

