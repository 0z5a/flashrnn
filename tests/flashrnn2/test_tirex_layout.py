"""Check TiRex's input/output recurrent axes independently of CUDA."""

import unittest

import torch

from flashrnn.flashrnn2.tirex_layout import recurrent_layout


class TiRexLayoutTest(unittest.TestCase):
    def test_recurrent_projection(self) -> None:
        hidden = torch.arange(1, 17, dtype=torch.float64).reshape(2, 2, 4)
        kernel = torch.arange(1, 129, dtype=torch.float64).reshape(2, 4, 16)
        official = torch.einsum("bhi,hio->bho", hidden, kernel)
        official = official.reshape(2, 2, 4, 4).permute(0, 2, 1, 3)
        candidate = torch.einsum("bhi,ghoi->bgho", hidden, recurrent_layout(kernel))
        torch.testing.assert_close(candidate, official, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
