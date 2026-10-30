import torch

# Past the recompile limit dynamo falls back to eager silently.
torch._dynamo.config.fail_on_recompile_limit_hit = True
