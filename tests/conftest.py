import torch

# Past dynamo's recompile limit a compiled function runs eager for the rest of
# the process; across the suite that would be silent. Make it an error.
torch._dynamo.config.fail_on_recompile_limit_hit = True
