# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.tools.logging import logger


class EarlyStoppingManager:
    """
    Manages early stopping based on an absolute validation loss threshold.

    Stops training after validation loss stays at or below a configured
    threshold for a configured number of consecutive validation checks.
    """
    
    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.0,
        enabled: bool = True,
    ):
        """
        Initialize the EarlyStoppingManager.

        Args:
            patience: Number of consecutive validation checks required at/below threshold.
            threshold: Absolute validation loss threshold.
            enabled: Whether early stopping is enabled
        """
        self.patience = max(1, int(patience))
        self.threshold = float(threshold)
        self.enabled = enabled

        self.below_threshold_counter = 0
    
    def check(self, validation_loss: float) -> bool:
        """
        Check if early stopping condition is met.

        Args:
            validation_loss: Current validation loss

        Returns:
            True if training should stop, False otherwise
        """
        if not self.enabled:
            return False

        if validation_loss <= self.threshold:
            self.below_threshold_counter += 1
            logger.info(
                f"[Early Stopping] Validation loss {validation_loss:.6f} <= threshold {self.threshold:.6f}. "
                f"Counter: {self.below_threshold_counter}/{self.patience}"
            )

            if self.below_threshold_counter >= self.patience:
                logger.warning(
                    f"[Early Stopping] Stopping training! "
                    f"Validation loss stayed <= {self.threshold:.6f} for "
                    f"{self.patience} consecutive validation checks."
                )
                return True

            return False

        self.below_threshold_counter = 0
        logger.info(
            f"[Early Stopping] Validation loss {validation_loss:.6f} > threshold {self.threshold:.6f}. "
            f"Counter reset to 0/{self.patience}"
        )
        return False
    
    def reset(self):
        """Reset the early stopping state."""
        self.below_threshold_counter = 0
