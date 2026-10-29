"""Compatibility imports for profile-aware policy models."""

from diffphys.models import ModifiedModel, VanillaModel, build_model

Model = VanillaModel


def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)
