"""The network that turns the model's impressions of a frame into a move.

Everything upstream of this file is frozen: the vision-language model is never trained, and the
reduction to 1024 dimensions is fixed once. This is the only part that learns, and it is small
-- about 79 million weights against the language model's seven billion.

It has to solve two problems that the frames themselves create.

**A frame does not arrive as one vector.** It arrives as a handful of tokens whose count changes
from frame to frame, because the model's answer runs to a different length each time. Something
has to compress a variable-length set into the fixed-size vector a recurrent network can accept,
and the paper's choice is a learned "CLS" token that attends over the frame's tokens and comes
away with a single summary (Section 3.2, Figure 2).

**One frame is not enough to decide anything.** The task is search. Standing in a doorway you
have already walked through looks exactly like standing in one you have not, and the right move
is opposite in the two cases. The difference is entirely in the past, so the summaries are read
in sequence by a recurrent network that carries what it has seen so far.

Every comparison in the paper -- prompted representations, unprompted ones, the image encoder
alone -- uses this same network unchanged, so that a difference in results can only come from
the representation (Appendix C.2, general item 5).
"""
from __future__ import annotations

import torch
from torch import nn

NUM_ACTIONS = 4          # stop, move forward, turn left, turn right (Appendix C.1)
NUM_OBJECTS = 6          # chair, bed, plant, toilet, tv_monitor, sofa (Section 4.2)

TOKEN_DIM = 2048         # PR2L: two layers of 1024 after the reduction (C.2, PR2L item 3).
                         # The image-encoder condition passes 2176 instead -- Listing 1 takes
                         # the width as `tf_embed_dim` precisely so it can differ.
MODEL_DIM = 1024         # Appendix C.2, PR2L item 5
FEEDFORWARD_DIM = 1024
NUM_HEADS = 1            # Appendix I, Listing 1 -- see `FrameSummary` for why this is 1 and not 8
DROPOUT = 0.1
SIDE_EMBED_DIM = 32      # per non-visual input, following PIRLNav section 3
LSTM_HIDDEN = 2048       # PIRLNav section 3; the paper says only "the same LSTM as [43]"
LSTM_LAYERS = 2


class FrameSummary(nn.Module):
    """Compress a frame's variable-length token set into one vector.

    A learned query token -- the paper calls it "CLS" -- is the decoder's only input. It attends
    over the frame's tokens and takes a weighted blend of them, and because the blend is over
    however many tokens arrive, the output size never changes. What the query attends to is
    learned: it can lean on the visual tokens when the target is in view and on the model's
    written reasoning when it is not.

    The padding mask is not optional. Batched frames are padded to a common length, and without
    the mask the query would average the padding in along with the real tokens, by an amount
    that varies with how much padding each frame happened to need.

    **One attention head, not several.** Appendix I's Listing 1 builds the layer as
    ``torch.nn.Transformer(1024, 1, num_encoder_layers=1, num_decoder_layers=1,
    dim_feedforward=1024, batch_first=True)``, and the second positional argument of that class
    is the head count. So the paper's query looks at the frame's tokens once across the full
    1024 dimensions rather than splitting them into narrower parallel views. ``num_heads`` is
    left adjustable because the alternative is worth trying later: with several heads, one can
    settle on the visual tokens and another on the model's written reasoning, instead of one
    blend having to serve both. Head count does not change the parameter count -- the projection
    is 1024 by 3072 either way -- only how the attention is partitioned.
    """

    def __init__(self, num_heads: int = NUM_HEADS, token_dim: int = TOKEN_DIM) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.project = nn.Linear(token_dim, MODEL_DIM)
        self.transformer = nn.Transformer(
            d_model=MODEL_DIM, nhead=num_heads,
            num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=FEEDFORWARD_DIM, dropout=DROPOUT,
            activation="relu", batch_first=True)
        self.query = nn.Embedding(1, MODEL_DIM)

    def forward(self, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        """[B, T, N, token_dim] and [B, T, N] -> [B, T, 1024]."""
        batch, steps, count, _ = tokens.shape
        flat = self.project(tokens.reshape(batch * steps, count, self.token_dim))
        mask = padding.reshape(batch * steps, count)

        # A frame can be *entirely* padding: batching trajectories of different lengths pads the
        # short ones out past their last real step. For those rows the query would attend over
        # nothing, and a softmax whose every position is forbidden is NaN -- not zero. The NaN
        # cannot be cleaned up afterwards either, because multiplying it by a validity flag of
        # zero leaves it NaN, so it reaches the loss and from there every weight in the network.
        # Leaving one position open keeps the arithmetic finite; the summary those rows produce
        # is meaningless, and every consumer already discards it by the step's validity flag.
        empty = mask.all(dim=1)
        if bool(empty.any()):
            mask = mask.clone()
            mask[empty, 0] = False

        query = self.query(torch.zeros(batch * steps, 1, dtype=torch.long,
                                       device=tokens.device))
        summary = self.transformer(flat, query,
                                   src_key_padding_mask=mask,
                                   memory_key_padding_mask=mask)
        return summary.reshape(batch, steps, MODEL_DIM)


class SideChannels(nn.Module):
    """Embed everything the agent knows that is not the picture.

    Appendix C.1 lists four such inputs beside the image: where the agent is, which way it
    faces, what it did last, and which object it is looking for. Habitat reports the first two
    relative to where the episode began, so "position" is a displacement and "compass" is a
    turn away from the starting heading.

    Each is widened to 32 numbers before being placed next to the 1024-wide frame summary.
    Handing over two or six raw numbers alongside a thousand would let them be ignored; the
    projection puts them on comparable footing. The width follows PIRLNav, which the paper
    inherits its policy from without restating the value.

    **How the heading is presented is a choice the paper leaves open.** Appendix C.1 describes
    the observation as a single float, and by default that single float is what the network
    reads, which is the literal reading and matches PIRLNav's description of passing the turn
    through a fully-connected layer. The alternative, selectable with
    ``heading_encoding="sincos"``, hands over the sine and cosine instead. The reason it is
    worth trying later: an angle just past pi and one just short of -pi describe almost the
    same direction but sit at opposite ends of the numeric range, so a network reading the raw
    value has to learn its way around that seam. Both forms carry the same information, so
    switching between them changes only how easily the network can use it.
    """

    def __init__(self, heading_encoding: str = "angle") -> None:
        super().__init__()
        if heading_encoding not in ("angle", "sincos"):
            raise ValueError(f"unknown heading encoding {heading_encoding!r}")
        self.heading_encoding = heading_encoding
        self.position = nn.Linear(2, SIDE_EMBED_DIM)
        self.heading = nn.Linear(1 if heading_encoding == "angle" else 2, SIDE_EMBED_DIM)
        self.previous_action = nn.Linear(NUM_ACTIONS, SIDE_EMBED_DIM)
        self.goal = nn.Linear(NUM_OBJECTS, SIDE_EMBED_DIM)

    @property
    def width(self) -> int:
        return 4 * SIDE_EMBED_DIM

    def forward(self, position: torch.Tensor, heading: torch.Tensor,
                previous_action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """position [B,T,2], heading [B,T], previous_action [B,T,4], goal [B,T,6]."""
        if self.heading_encoding == "angle":
            facing = heading.unsqueeze(-1)
        else:
            facing = torch.stack([torch.sin(heading), torch.cos(heading)], dim=-1)
        return torch.cat([self.position(position), self.heading(facing),
                          self.previous_action(previous_action), self.goal(goal)], dim=-1)


class NavigationPolicy(nn.Module):
    """Frame summary and side channels, read in sequence, turned into an action."""

    def __init__(self, heading_encoding: str = "angle", num_heads: int = NUM_HEADS,
                 token_dim: int = TOKEN_DIM) -> None:
        super().__init__()
        self.summary = FrameSummary(num_heads=num_heads, token_dim=token_dim)
        self.side = SideChannels(heading_encoding=heading_encoding)
        self.memory = nn.LSTM(MODEL_DIM + self.side.width, LSTM_HIDDEN,
                              num_layers=LSTM_LAYERS, batch_first=True)
        self.action = nn.Linear(LSTM_HIDDEN, NUM_ACTIONS)

    def forward(self, tokens: torch.Tensor, padding: torch.Tensor,
                position: torch.Tensor, heading: torch.Tensor,
                previous_action: torch.Tensor, goal: torch.Tensor,
                state: tuple[torch.Tensor, torch.Tensor] | None = None):
        """Return action logits [B, T, 4] and the recurrent state to carry forward.

        Trajectories are fed whole during training, so the recurrent state starts empty and is
        rolled honestly from the first step. During evaluation the agent takes one step at a
        time and hands the returned state back on the next call.
        """
        summary = self.summary(tokens, padding)
        combined = torch.cat([summary, self.side(position, heading, previous_action, goal)],
                             dim=-1)
        recurrent, state = self.memory(combined, state)
        return self.action(recurrent), state


def count_parameters(module: nn.Module) -> dict[str, int]:
    """Parameter counts, used as a check that the network is the size it should be."""
    parts = {name: sum(p.numel() for p in child.parameters())
             for name, child in module.named_children()}
    parts["total"] = sum(p.numel() for p in module.parameters())
    return parts


if __name__ == "__main__":
    for encoding in ("angle", "sincos"):
        for heads in (1, 8):
            policy = NavigationPolicy(heading_encoding=encoding, num_heads=heads)
            counts = count_parameters(policy)
            print(f"heading_encoding={encoding} num_heads={heads}")
            for name, value in counts.items():
                print(f"  {name:10s} {value:>12,}")

    batch, steps, tokens = 2, 5, 80
    logits, state = policy(
        torch.randn(batch, steps, tokens, TOKEN_DIM),
        torch.zeros(batch, steps, tokens, dtype=torch.bool),
        torch.randn(batch, steps, 2), torch.randn(batch, steps),
        torch.zeros(batch, steps, NUM_ACTIONS), torch.zeros(batch, steps, NUM_OBJECTS))
    print(f"\n  logits {tuple(logits.shape)}  hidden {tuple(state[0].shape)}")
