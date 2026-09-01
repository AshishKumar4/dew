"""Clipping and range conversion for image tensors."""

import jax.numpy as jnp


def clip_images(images, clip_min=-1, clip_max=1):
    """Clip image values to a specified range.
    
    Args:
        images: Images to clip
        clip_min: Minimum value
        clip_max: Maximum value
    
    Returns:
        Clipped images
    """
    return jnp.clip(images, clip_min, clip_max)

def denormalize_images(images, target_type=jnp.uint8, source_range=(-1, 1), target_range=(0, 255)):
    """Convert images from normalized range (e.g. [-1, 1]) to target range (e.g. [0, 255]).
    
    Args:
        images: Normalized images
        target_type: Target dtype (e.g. jnp.uint8 for standard images)
        source_range: Tuple of (min, max) for the source normalization range
        target_range: Tuple of (min, max) for the target range
        
    Returns:
        Denormalized images in the target dtype
    """
    src_min, src_max = source_range
    tgt_min, tgt_max = target_range
    
    # First clip to ensure we're in the expected source range
    images = clip_images(images, src_min, src_max)
    
    # Scale to [0, 1]
    images = (images - src_min) / (src_max - src_min)
    
    # Scale to target range
    images = images * (tgt_max - tgt_min) + tgt_min
    
    # Convert to target dtype if needed
    if target_type is not None:
        images = images.astype(target_type)
    
    return images
