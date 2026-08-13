from fedpareto.aggregation.fedavg import fedavg_aggregate
from fedpareto.aggregation.robust import coord_median_aggregate, trimmed_mean_aggregate, krum_aggregate
from fedpareto.aggregation.fltrust import fltrust_aggregate
from fedpareto.aggregation.fedpareto import fedpareto_aggregate

__all__ = [
    "fedavg_aggregate",
    "coord_median_aggregate",
    "trimmed_mean_aggregate",
    "krum_aggregate",
    "fltrust_aggregate",
    "fedpareto_aggregate",
]
