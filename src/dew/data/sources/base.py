from abc import ABC, abstractmethod
import grain.python as pygrain
from typing import Dict, Any, Callable, List, Optional
import jax.numpy as jnp
from functools import partial


class DataSource(ABC):
    """Base class for all data sources in Dew."""
    
    @abstractmethod
    def get_source(self, path_override: str) -> Any:
        """Return the data source object.
        
        Args:
            path_override: Path to the dataset, overriding the default.
            
        Returns:
            A data source object compatible with grain or other loaders.
        """
        pass


class DataAugmenter(ABC):
    """Base class for all data augmenters in Dew.

    The contract is deliberately only `create_transform`: no grain pipeline in
    this repo applies a filter operation, so filtering lives solely on the one
    augmenter that has a working implementation (`ImageGCSAugmenter.create_filter`,
    reachable through the legacy `gcs_filters` helper).
    """
    
    @abstractmethod
    def create_transform(self, **kwargs) -> Callable[[], pygrain.Transformation]:
        """Create a transformation function for the data.
        
        Args:
            **kwargs: Additional arguments for the transformation.
            
        Returns:
            A callable that returns a pygrain.Transformation instance.
        """
        pass


class MediaDataset:
    """A class combining a data source and an augmenter for a complete dataset."""
    
    def __init__(self, 
                 source: DataSource, 
                 augmenter: DataAugmenter,
                 media_type: str = "image"):
        """Initialize a MediaDataset.
        
        Args:
            source: The data source.
            augmenter: The data augmenter.
            media_type: Type of media ("image", "video", etc.)
        """
        self.source = source
        self.augmenter = augmenter
        self.media_type = media_type
    
    def get_source(self, path_override: str) -> Any:
        """Get the data source.
        
        Args:
            path_override: Path to override the default data source path.
            
        Returns:
            A data source object.
        """
        return self.source.get_source(path_override)
    
    def get_augmenter(self, **kwargs) -> Callable[[], pygrain.Transformation]:
        """Get the augmenter transformation.
        
        Args:
            **kwargs: Additional arguments for the augmenter.
            
        Returns:
            A callable that returns a pygrain.Transformation instance.
        """
        return self.augmenter.create_transform(**kwargs)