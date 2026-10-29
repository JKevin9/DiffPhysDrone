"""Physics-clock inference scheduling, independent of simulator/manager cadence."""

import torch

from .perception import decode_action, policy_state


class PolicyRuntime:
    def __init__(self, model, profile, num_envs, device):
        self.model, self.profile = model.eval(), profile
        self.device = torch.device(device)
        self.hidden = torch.zeros((num_envs, profile.hidden_dim), device=device)
        self.acceleration = torch.zeros((num_envs, 3), device=device)
        self.next_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.inference_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_update_step = torch.full_like(self.next_step, -1)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.hidden[ids] = 0
        self.acceleration[ids] = 0
        self.next_step[ids] = 0
        self.fresh[ids] = True
        self.last_update_step[ids] = -1

    @torch.inference_mode()
    def update(self, physics_step, image_provider, position, velocity, rotation, target, speed, completed=None):
        due = physics_step >= self.next_step
        if completed is not None:
            self.acceleration[completed] = 0
            self.hidden[completed] = 0
            due &= ~completed
        ids = due.nonzero().flatten()
        if ids.numel() == 0:
            return self.acceleration, False
        image = image_provider()
        expected = (self.hidden.shape[0], 1, self.profile.image_height, self.profile.image_width)
        if tuple(image.shape) != expected or image.device != self.hidden.device or not torch.isfinite(image).all():
            raise ValueError(f"Invalid {self.profile.name} image shape/device/values: {tuple(image.shape)}")
        state, frame, _ = policy_state(position, velocity, rotation, target, speed, self.profile.safety_margin)
        if state.shape != (self.hidden.shape[0], 10) or not torch.isfinite(state).all():
            raise ValueError("Invalid 10-dimensional Newton state")
        action, _, hidden = self.model(image[ids], state[ids], self.hidden[ids])
        if action.shape != (len(ids), 6) or hidden.shape != (len(ids), 192):
            raise ValueError("Invalid Newton policy output shape")
        acceleration, _ = decode_action(action, frame[ids])
        if not torch.isfinite(acceleration).all() or not torch.isfinite(hidden).all():
            raise ValueError("Non-finite Newton policy output")
        first = self.fresh[ids]
        # Retain the Newton reset padding: discard both first action and hidden.
        acceleration[first] = 0
        hidden[first] = 0
        self.acceleration[ids], self.hidden[ids] = acceleration, hidden
        self.fresh[ids] = False
        self.next_step[ids] = physics_step + self.profile.benchmark_period_steps
        self.last_update_step[ids] = physics_step
        self.inference_count[ids] += 1
        return self.acceleration, True
