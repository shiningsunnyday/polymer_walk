from torch_geometric.nn.conv import GINEConv, GINConv, GATConv, GCNConv
from typing import Callable, Union
from torch import Tensor
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size


class MyGINConv(GINEConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)
