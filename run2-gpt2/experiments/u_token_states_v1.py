"""One isolated period-eight writer on linearly adjusted recorded attention outputs."""
from pathlib import Path
import torch
from n_token_states_v1 import reconstruct as writer_piece
from u_recorded_writer_inputs_v1 import reconstruct as writer_inputs
ROOT=Path(__file__).resolve().parents[1]

def reconstruct(host,delta,data):
    geometry=torch.load(ROOT/'artifacts/u_period8_coordinates_v1.pt',weights_only=True)
    inputs=writer_inputs(host,geometry,data)
    return writer_piece(host,delta,inputs)
