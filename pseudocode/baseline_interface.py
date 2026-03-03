"""
Pseudocode: Baseline Model Interface

This file documents the common interface for all baseline power forecasting models.
All baseline models inherit from BaseSFTModel and follow a unified input/output specification.

Purpose:
    - Define standard interface for baseline models
    - Ensure consistent input/output formats
    - Enable fair comparison across different methods
"""

# ===================================================================
# 1. Base Model Arguments
# ===================================================================

class BaseSFTModelArguments:
    """
    Base class for baseline model hyperparameters.

    All baseline models should define their own Arguments class
    that inherits from this base class.

    Common Parameters:
        input_weather_past_len: 120 (5 days hourly)
        input_weather_future_len: 120 (5 days hourly)
        input_weather_dim: 12 (solar) or 15 (wind)
        input_power_past_len: 480 (5 days @ 15-min intervals)
        output_power_len: 480 (5 days @ 15-min intervals)
        task_type: "wind" or "solar"
    """

    def __init__(self, task_type="wind"):
        """
        Parameters:
            task_type: "wind" or "solar"
        """
        self.task_type = task_type

        # Input dimensions (fixed for all baselines)
        self.input_weather_past_len = 120
        self.input_weather_future_len = 120
        self.input_power_past_len = 480
        self.output_power_len = 480

        # Weather dimension depends on task type
        if task_type == "wind":
            self.input_weather_dim = 15  # Wind-specific weather variables
        else:  # solar
            self.input_weather_dim = 12  # Solar-specific weather variables

    @classmethod
    def get_test_configs(cls):
        """
        Return 3 hyperparameter configurations for testing.

        Each baseline model should override this method to provide
        3 different configurations for hyperparameter search.

        Returns:
            list of 3 model argument instances

        Example:
            return [
                cls(hidden_dim=512, num_layers=3),  # Config 1
                cls(hidden_dim=256, num_layers=3),  # Config 2
                cls(hidden_dim=512, num_layers=2),  # Config 3
            ]
        """
        raise NotImplementedError("Subclasses must implement get_test_configs()")


# ===================================================================
# 2. Base Model Class
# ===================================================================

class BaseSFTModel:
    """
    Base class for all baseline power forecasting models.

    All baseline models must:
    1. Inherit from this class
    2. Implement _forward_impl() method
    3. Follow the standard input/output specification

    Standard Interface:
        Inputs:
            input_weather_past: [B, 120, 12/15]
            input_weather_future: [B, 120, 12/15]
            input_power_past: [B, 480]

        Output:
            output_power: [B, 480]

        Training Output:
            {"loss": scalar, "output": [B, 480]}
    """

    def __init__(self, args: BaseSFTModelArguments):
        """
        Initialize baseline model.

        Parameters:
            args: Model hyperparameters
        """
        self.args = args

        # Store input dimensions for convenience
        self.input_weather_past_len = args.input_weather_past_len
        self.input_weather_future_len = args.input_weather_future_len
        self.input_weather_dim = args.input_weather_dim
        self.input_power_past_len = args.input_power_past_len
        self.output_power_len = args.output_power_len
        self.task_type = args.task_type

    def forward(self, input_weather_past, input_weather_future, input_power_past, target_power=None):
        """
        Forward pass with optional training mode.

        Inputs:
            input_weather_past: [B, 120, 12/15]
                - Historical weather (5 days hourly)
                - 12 channels for solar, 15 for wind

            input_weather_future: [B, 120, 12/15]
                - Future weather forecast (5 days hourly)
                - Same channel structure as past

            input_power_past: [B, 480]
                - Historical power (5 days @ 15-min intervals)

            target_power: [B, 480] (optional)
                - Target power for training
                - If provided, returns loss

        Returns:
            If target_power is None (inference):
                output_power: [B, 480]

            If target_power is provided (training):
                {
                    "loss": scalar,
                    "output": [B, 480]
                }

        Implementation:
            # Call subclass implementation
            # output = self._forward_impl(input_weather_past, input_weather_future, input_power_past)

            # If training mode
            # if target_power is not None:
            #     loss = MSE(output, target_power)
            #     return {"loss": loss, "output": output}
            # else:
            #     return output
        """
        pass

    def _forward_impl(self, input_weather_past, input_weather_future, input_power_past):
        """
        Model-specific forward implementation.

        Subclasses MUST implement this method.

        Inputs:
            input_weather_past: [B, 120, 12/15]
            input_weather_future: [B, 120, 12/15]
            input_power_past: [B, 480]

        Returns:
            output_power: [B, 480]

        Example Implementation:
            # Concatenate inputs
            # weather = concatenate([input_weather_past, input_weather_future], dim=1)  # [B, 240, 12/15]

            # Process weather
            # weather_features = self.weather_encoder(weather)  # [B, hidden_dim]

            # Process power
            # power_features = self.power_encoder(input_power_past)  # [B, hidden_dim]

            # Combine and predict
            # combined = concatenate([weather_features, power_features], dim=1)
            # output = self.decoder(combined)  # [B, 480]

            # return output
        """
        raise NotImplementedError("Subclasses must implement _forward_impl()")


# ===================================================================
# 3. Loss Function
# ===================================================================

def compute_loss(predictions, targets):
    """
    Compute Mean Squared Error loss.

    Formula:
        MSE = (1/N) * Σ(predictions - targets)²

    Inputs:
        predictions: [B, 480]
        targets: [B, 480]

    Returns:
        loss: scalar

    Implementation:
        # loss = mean((predictions - targets) ** 2)
        # return loss
    """
    pass


# ===================================================================
# 4. Weather Variables
# ===================================================================

"""
Weather Variables by Task Type:

Wind (15 channels):
    1. u10: 10m U-component of wind (m/s)
    2. v10: 10m V-component of wind (m/s)
    3. u100: 100m U-component of wind (m/s)
    4. v100: 100m V-component of wind (m/s)
    5. u200: 200m U-component of wind (m/s)
    6. v200: 200m V-component of wind (m/s)
    7. t2m: 2m temperature (K)
    8. sp: Surface pressure (Pa)
    9. tcc: Total cloud cover (0-1)
    10. tp: Total precipitation (m)
    11. z1000: 1000 hPa geopotential (m²/s²)
    12. q1000: 1000 hPa specific humidity (kg/kg)
    13. t1000: 1000 hPa temperature (K)
    14. u1000: 1000 hPa U-component of wind (m/s)
    15. v1000: 1000 hPa V-component of wind (m/s)

Solar (12 channels):
    1. u10: 10m U-component of wind (m/s)
    2. v10: 10m V-component of wind (m/s)
    3. t2m: 2m temperature (K)
    4. sp: Surface pressure (Pa)
    5. ssr: Surface solar radiation (J/m²)
    6. tcc: Total cloud cover (0-1)
    7. tp: Total precipitation (m)
    8. z1000: 1000 hPa geopotential (m²/s²)
    9. q1000: 1000 hPa specific humidity (kg/kg)
    10. t1000: 1000 hPa temperature (K)
    11. u1000: 1000 hPa U-component of wind (m/s)
    12. v1000: 1000 hPa V-component of wind (m/s)

Note: Weather variables are averaged over the 8×8 spatial patch
"""


# ===================================================================
# 5. Example: Simple MLP Baseline
# ===================================================================

class ExampleMLPArguments(BaseSFTModelArguments):
    """
    Arguments for Example MLP baseline.
    """

    def __init__(self, hidden_dim=256, num_layers=3, task_type="wind"):
        super().__init__(task_type)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    @classmethod
    def get_test_configs(cls):
        """Return 3 configurations for hyperparameter search."""
        return [
            cls(hidden_dim=512, num_layers=3),  # Config 1
            cls(hidden_dim=256, num_layers=3),  # Config 2
            cls(hidden_dim=512, num_layers=2),  # Config 3
        ]


class ExampleMLP(BaseSFTModel):
    """
    Simple MLP baseline for power forecasting.

    Architecture:
        1. Flatten and concatenate all inputs
        2. Pass through multi-layer perceptron
        3. Output power predictions

    This is a simple baseline that ignores temporal structure.
    """

    def __init__(self, args: ExampleMLPArguments):
        super().__init__(args)

        # Calculate input dimension
        # weather_past: 120 * 12/15
        # weather_future: 120 * 12/15
        # power_past: 480
        input_dim = (self.input_weather_past_len * self.input_weather_dim +
                     self.input_weather_future_len * self.input_weather_dim +
                     self.input_power_past_len)

        # Build MLP layers
        # self.layers = []
        # current_dim = input_dim
        # for i in range(args.num_layers - 1):
        #     self.layers.append(Linear(current_dim, args.hidden_dim))
        #     self.layers.append(ReLU())
        #     current_dim = args.hidden_dim
        # self.layers.append(Linear(current_dim, self.output_power_len))

    def _forward_impl(self, input_weather_past, input_weather_future, input_power_past):
        """
        Forward pass for MLP baseline.

        Inputs:
            input_weather_past: [B, 120, 12/15]
            input_weather_future: [B, 120, 12/15]
            input_power_past: [B, 480]

        Returns:
            output_power: [B, 480]

        Implementation:
            # Flatten weather inputs
            # weather_past_flat = flatten(input_weather_past)  # [B, 120*12/15]
            # weather_future_flat = flatten(input_weather_future)  # [B, 120*12/15]

            # Concatenate all inputs
            # x = concatenate([weather_past_flat, weather_future_flat, input_power_past], dim=1)

            # Pass through MLP
            # for layer in self.layers:
            #     x = layer(x)

            # return x  # [B, 480]
        """
        pass


# ===================================================================
# 6. Training Workflow
# ===================================================================

def train_baseline_model(model, train_dataset, valid_dataset, config):
    """
    Train a baseline model on station-specific data.

    Process:
        1. Initialize optimizer and scheduler
        2. For each epoch:
           a. For each batch:
              - Forward pass
              - Compute loss
              - Backward pass
              - Update weights
           b. Evaluate on validation set
           c. Save checkpoint if best
        3. Return best model

    Configuration:
        optimizer: AdamW
        learning_rate: 1e-4
        weight_decay: 0.01
        batch_size: 32
        num_epochs: 10
        scheduler: Cosine annealing

    Returns:
        trained_model: Best model based on validation loss
    """
    pass


# ===================================================================
# 7. Evaluation Workflow
# ===================================================================

def evaluate_baseline_model(model, test_dataset, station_id):
    """
    Evaluate baseline model on test set.

    Process:
        1. Load model checkpoint
        2. For each test sample:
           - Run inference
           - Collect predictions
        3. Compute metrics
        4. Save results

    Metrics:
        - RMSE: Root Mean Square Error
        - MAE: Mean Absolute Error
        - R²: Coefficient of Determination
        - MAPE: Mean Absolute Percentage Error

    Returns:
        results = {
            "station_id": station_id,
            "rmse": float,
            "mae": float,
            "r2": float,
            "mape": float,
        }
    """
    pass
