### **1. Previous Conversation**

The conversation began with a high-level strategic request to compare two knowledge distillation techniques for a Deep Reinforcement Learning (DRL) financial trading agent: Transformed Teacher Matching (TTM), a logits-based method, and Probabilistic Knowledge Transfer (PKT), a feature-based method. I, as the Master Engineer, provided an initial architectural assessment that TTM's focus on policy regularization would likely be superior to PKT's feature-space mimicry in a noisy financial domain.

This led to a multi-stage process. First, we established a rigorous academic evaluation framework, defining key metrics across Financial Performance, RL Dynamics, and Behavioral Statistics. Then, the conversation evolved into a deep, collaborative debugging and refactoring effort of the user's experimental framework. We systematically identified and fixed a series of architectural flaws and bugs, including:
*   A `TypeError` caused by a base experiment class (`MarketExperiment5`) having mixed responsibilities (training and evaluation) and returning a complex, nested data structure.
*   An `UnboundLocalError` from leftover code after the initial refactoring.
*   Incorrect data logging that resulted in empty training dynamics reports.
*   An `AttributeError` in the reporting pipeline due to an API mismatch.

Through this process, we established a robust, three-tier architecture: an **Orchestrator** for running experiments, an **Analysis Engine** (`AcademicIntegration`) for calculating metrics, and a **Reporting Engine** (`AcademicEvaluator`) for formatting the final output.

Finally, after analyzing the initial, divergent results between TTM and PKT, the conversation returned to strategy. Citing the user's own thesis and other academic papers, we concluded that the most promising path forward was not to abandon PKT, but to enhance it by creating a hybrid distillation method that combines PKT's feature-based loss with a logits-based loss for regularization.

### **2. Current Work**

The immediate task is the implementation of this **hybrid distillation method**. We are modifying the existing `offline_pkt` experiment, which is managed by the `MarketExperiment5` class, to incorporate a logits-based loss component. The goal is to test the hypothesis that adding a policy regularization signal (from the logits) can improve the poor generalization performance of the purely feature-based PKT method. This involves reusing a proven logits-loss pattern from another of the user's experiment files (`MarketExperiment4`) to ensure robustness and consistency.

### **3. Key Technical Concepts**

*   **Knowledge Distillation (KD):** The core technique of transferring knowledge from a teacher model to a student model.
*   **Logits-based Distillation:** Methods that match the final policy outputs of the models. We are using a standard **Cross-Entropy Loss** on the logits for the new hybrid component.
*   **Feature-based Distillation:** Methods that match intermediate representations. We are using **Probabilistic Knowledge Transfer (PKT)**, which matches the geometry of the feature space.
*   **Hybrid Distillation:** The current approach, which combines both logits-based and feature-based losses in a single objective function: `L_distill = β_pkt * L_PKT + β_logit * L_Logit`.
*   **Offline Distillation:** The framework we are using, where the student learns from a static, pre-trained ensemble of teacher models.
*   **Separation of Concerns:** The guiding architectural principle for our refactoring, ensuring that training, analysis, and reporting are handled by distinct, specialized modules.
*   **Python/PyTorch:** The implementation uses Python 3.8, PyTorch, and common data science libraries. Key functions include `torch.nn.functional.cross_entropy`.

### **4. Relevant Files and Code**

*   **`tsrl/experiments/market/experiment_offline_pkt.py` (`MarketExperiment5`)**
    *   **Importance:** This is the primary file being modified. It is the base class for the PKT experiment and will house the new hybrid loss logic.
    *   **Changes:** We will modify its `_compute_distillation_loss` method to calculate both the PKT loss and the logits loss. We will also update its public `student_train` adapter method to accept and set the new hyperparameters (`pkt_weight`, `logit_weight`).
    *   **Important Code Snippet (Target Implementation):**
        ```python
        # In MarketExperiment5._compute_distillation_loss
        
        # ... (get student/teacher features and logits) ...

        # --- 1. PKT Loss (Feature-based) ---
        pkt_loss = prob_loss(...)

        # --- 2. Logits-based Loss (Pattern from MarketExperiment4) ---
        logit_loss = F.cross_entropy(student_logits, ensemble_teacher_logits)

        # --- 3. Combine losses with configurable weights ---
        combined_loss = (self.pkt_weight * pkt_loss) + (self.logit_weight * logit_loss)
        
        return combined_loss
        ```

*   **`experiment_config_ttm_pkt.yaml`**
    *   **Importance:** This file controls all experiment hyperparameters.
    *   **Changes:** We will add `pkt_weight` and `logit_weight` to the `training_params` section for the `offline_pkt` experiment to control the new hybrid loss.

*   **`train_rl_experiments_ttm_pkt.py` (Orchestrator)**
    *   **Importance:** This is the main entry point that runs the experiments.
    *   **Changes:** No further changes are needed here, as our previous refactoring has already prepared it to handle different experiment configurations.

*   **`experiment_code/academic_evaluator.py` & `experiment_code/academic_integration.py`**
    *   **Importance:** These modules handle all post-run analysis and report generation.
    *   **Changes:** No further changes are needed. They are architecturally sound and will correctly process the results from the new hybrid experiment.

### **5. Problem Solving**

*   **Solved:** We have systematically resolved a series of critical bugs and architectural flaws, including `TypeError` from nested data structures, `UnboundLocalError` from leftover code, and incorrect data logging, leading to a robust and reliable experimental framework.
*   **Ongoing Troubleshooting:** The current task is not troubleshooting a bug, but rather addressing a research problem: the poor generalization of the PKT method. Our proposed solution is to implement the PKT+Logits hybrid to see if it mitigates this issue.

### **6. Pending Tasks and Next Steps**

The path forward is clear and consists of implementing the sprint handoff for the hybrid PKT+Logits method.

*   **Next Step 1: Implement the Hybrid Loss Calculation**
    *   **Task:** Modify the `_compute_distillation_loss` method in `tsrl/experiments/market/experiment_offline_pkt.py`.
    *   **Verbatim Instruction:** "Replace the entire method with the following definitive implementation, which now calculates both losses." (from previous response)

*   **Next Step 2: Update the `student_train` Adapter**
    *   **Task:** Modify the public `student_train` method in `tsrl/experiments/market/experiment_offline_pkt.py` to handle the new hyperparameters.
    *   **Verbatim Instruction:** "Add the logic to set the instance variables for the weights." (from previous response)

*   **Next Step 3: Update the YAML Configuration**
    *   **Task:** Add the `pkt_weight` and `logit_weight` parameters to the `offline_pkt` experiment block in `experiment_config_ttm_pkt.yaml`.
    *   **Verbatim Instruction:** "Update the `offline_pkt` section of your config file." (from previous response)

*   **Final Step: Execute and Analyze**
    *   **Task:** Run the full comparative suite (`Baseline`, `TTM`, and the new `PKT+Logits Hybrid`).
    *   **Goal:** Analyze the generated academic reports to determine if the hybrid approach successfully improved the generalization performance of the PKT method compared to both the original PKT and the pure TTM method.


Of course. To ensure the next agent has complete and seamless context, I will generate a self-contained "project snapshot." This will include the full code for all critical files in their current, corrected state, along with architectural summaries. This package is designed to be the definitive source of truth for continuing our work.

The immediate next step, as we've established, is to implement the **PKT+Logits Hybrid** distillation method by modifying the `MarketExperiment5` class. The following files provide all the necessary context and components to execute that task.

---
### **File 1: The Configuration File**
*   **Path:** `experiment_config_ttm_pkt.yaml`
*   **Role:** The central control file for the entire framework. It defines the experiments to be run, the model architecture, and all hyperparameters.
*   **Master Engineer's Note:** The `offline_pkt` method has been prepared for the upcoming hybrid experiment. Note the addition of `pkt_weight` and `logit_weight`, which will control the new hybrid loss function.

```yaml
# The central configuration file for all experiments.

paths:
  experiments: "saved_models_experiments"
  figures: "thesis_results/figures"
  curve_data: "thesis_results/curve_data"
  distillation_names:
    - "no_distillation"
    - "rl_ttm"
    - "offline_pkt"

data:
  train_start: "2018-01-01"
  train_end: "2021-03-14"
  test_start: "2021-03-15"
  test_end: "2022-02-01"
  features:
    # ... (full feature list as provided previously) ...
    - {name: "int_bar_changes", func_name: "inter_bar_changes", columns: ["close", "high", "low"], use_pct: true}
    - {name: "int_bar_changes_10", func_name: "inter_bar_changes", columns: ["close", "high", "low"], use_pct: true, smoothing_window: 10}
    - {name: "internal_bar_diff", func_name: "internal_bar_diff", use_pct: true}
    - {name: "hl_to_pclose", func_name: "hl_to_pclose"}
    - {name: "hlvol_10", func_name: "hl_volatilities", smoothing_window: 10}
    - {name: "hlvol_50", func_name: "hl_volatilities", smoothing_window: 50}
    - {name: "rvol_10", func_name: "return_volatilities", smoothing_window: 10}
    - {name: "rvol_50", func_name: "return_volatilities", smoothing_window: 50}
    - {name: "time_features_day", func_name: "time_feature_day"}
    - {name: "time_features_year", func_name: "time_feature_year"}
    - {name: "time_features_month", func_name: "time_feature_month"}
    - {name: "time_features_week", func_name: "time_feature_week"}

model:
  combine_policy_value: false
  nb_actions: 3
  lstm_size: 32
  actor_size: [32]
  critic_size: [32]
  dropout: 0.2

training:
  num_runs: 1 # Set to 1 for debugging, increase for final experiments
  env_params:
    max_episode_steps: 40
    commission_punishment: 0.0002

teacher_config: # Default teacher config, can be overridden per method
  teacher_save_dir: "shared_teachers"
  teacher_pool_size: 5
  n_teachers_to_select: 5
  base_seed: 42
  teacher_epochs: 500

distillation:
  - method: "no_distillation"
    display_name: "Baseline"
    training_params:
      n_epochs: 500
      batch_size: 32
      lr: 0.0005

  - method: "rl_ttm"
    display_name: "TTM"
    training_params:
      n_epochs: 500
      batch_size: 32
      lr: 0.0005
      ttm_weight: 1.0
      ttm_l: 1.0

  - method: "offline_pkt"
    display_name: "PKT + Logits"
    training_params:
      n_epochs: 500
      batch_size: 32
      lr: 0.0005
      # --- HYBRID LOSS WEIGHTS ---
      pkt_weight: 0.5   # Weight for the feature-based PKT loss
      logit_weight: 0.5 # Weight for the logits-based KD loss
      kd_temperature: 4.0 # Temperature for the logits loss component
      # --- PKT KERNEL PARAMS ---
      kernel_parameters:
        teacher: 'cosine'
        student: 'cosine'
        loss: 'kl'
```

---
### **File 2: The Main Orchestrator**
*   **Path:** `train_rl_experiments_ttm_pkt.py`
*   **Role:** The main entry point. It reads the config, manages the experiment lifecycle, and initiates the final analysis and reporting pipeline.
*   **Master Engineer's Note:** This version is fully refactored. Note the clean `_execute_single_run` method, which now explicitly controls the **Train -> Evaluate -> Backtest** sequence. Its final action is a single call to `academic_integration.generate_academic_report`.

```python
# This is a condensed version of the orchestrator showing the key logic.
# Assume all necessary imports are present.

class TTMPKTComparisonOrchestrator:
    def __init__(self, config_path: str, use_training: bool = True):
        # ... (initialization as provided before) ...
        self.academic_integration = AcademicIntegration(self)

    @staticmethod
    def _get_experiment_class(method_name: str):
        if method_name == 'no_distillation':
            from tsrl.experiments.market.experiment_no_distillation import MarketExperiment1
            return MarketExperiment1
        if method_name == 'rl_ttm':
            return RlTtmExperiment
        elif method_name == 'offline_pkt':
            return MarketExperiment5
        else:
            raise ValueError(f"Unknown method: {method_name}")

    def run_experiments(self, ...):
        # ... (logic for handling baseline and preparing teachers) ...
        
        all_results = {}
        for method_config in self.config['distillation']:
            # ... (loop and skip logic) ...
            
            method_results, agg_results = self.run_method_experiments(...)
            all_results[method_name] = {'results': method_results, 'aggregates': agg_results}

        # FINAL HANDOFF
        print("\nGenerating comprehensive academic evaluation reports...")
        try:
            academic_files = self.academic_integration.generate_academic_report(all_results)
            # ... (print file paths) ...
        except Exception as e:
            print(f"⚠️ Academic report generation failed: {e}")

    def run_method_experiments(self, ...):
        # ... (logic to loop through num_runs) ...
        for run_idx in range(num_runs):
            run_results = self._execute_single_run(...)
            # ... (aggregate and save run results) ...

    def _execute_single_run(self, method_name: str, ...):
        """
        Executes a single, full experimental run with a clear, decoupled
        Train -> Evaluate -> Backtest sequence.
        """
        exp_class = self._get_experiment_class(method_name)
        model_params = self.config['model'].copy()
        model_params['num_inputs'] = list(self.data['feature_dict'].values())[0].shape[1]
        experiment = exp_class(...)

        # === STEP 1: TRAIN THE MODEL ===
        training_metrics = {}
        if self.use_training:
            # ... (dispatch to experiment.train or experiment.student_train) ...

        # === STEP 2: EVALUATE THE TRAINED MODEL ===
        eval_dict = experiment.eval(self.data)

        # === STEP 3: BACKTEST AND CALCULATE PERFORMANCE METRICS ===
        detailed_pnls = experiment.detailed_backtest(...)
        final_performance_report = experiment.backtest(...)

        # === COMPILE FINAL RESULTS FOR THIS RUN ===
        # ...
        return { ... } # The final, clean run dictionary
```

---
### **File 3: The Base Experiment Class (Target for Modification)**
*   **Path:** `tsrl/experiments/market/experiment_offline_pkt.py`
*   **Role:** The parent class for all offline distillation methods. It contains the core PPO training loop (`_student_train_internal`) and the public adapter (`student_train`).
*   **Master Engineer's Note:** This is the file where the next agent will work. The immediate task is to modify `_compute_distillation_loss` to implement the hybrid PKT+Logits logic, using the new hyperparameters passed down through `training_params`.

```python
# Assumes all necessary imports are present.

class MarketExperiment5(TorchExperiment):
    # ... (__init__, eval, backtest, etc.) ...

    def _compute_distillation_loss(self, student_eval: Dict, teacher_eval: Dict, **kwargs) -> torch.Tensor:
        """
        This is the hook for PKT. The NEXT TASK is to modify this method
        to calculate a hybrid PKT + Logits loss.
        """
        student_features = kwargs.get("student_features")
        # ... (logic to get teacher features) ...
        ensemble_teacher_features = torch.stack(teacher_features_list).mean(dim=0)
        
        # Current PKT-only implementation
        pkt_loss = prob_loss(
            teacher_features=ensemble_teacher_features,
            student_features=student_features,
            kernel_parameters=kwargs.get('kernel_parameters', {})
        )
        return pkt_loss

    def student_train(self, data, model_params=None, env_params=None, **training_params):
        """
        PUBLIC-FACING training method. Sets up weights and delegates to the internal loop.
        """
        # Set up weights for the hybrid loss calculation
        self.pkt_weight = training_params.get('pkt_weight', 1.0)
        self.logit_weight = training_params.get('logit_weight', 1.0)
        
        training_report = self._student_train_internal(data, model_params, env_params, **training_params)

        structured_metrics = {
            'final_train_metrics': training_report.get('final_train_metrics', {}),
            'model_stats': training_report.get('model_stats', {}),
            'total_epochs': training_report.get('total_epochs', 0),
            'method': 'Offline_PKT_Hybrid',
            'distillation_type': 'hybrid_features_and_logits',
            'pkt_weight': self.pkt_weight,
            'logit_weight': self.logit_weight,
        }
        return structured_metrics

    def _student_train_internal(self, data, model_params=None, env_params=None, **training_params):
        """
        INTERNAL training method. Contains the full PPO loop and returns
        a dictionary of comprehensive training metrics and history lists.
        """
        # ... (full, corrected PPO training loop as defined in previous steps) ...
        # This method is now stable and correctly logs all history lists.
```

---
### **File 4: The TTM Experiment Class (Reference Implementation)**
*   **Path:** `tsrl/experiments/market/experiment_rl_ttm.py`
*   **Role:** The child class for the TTM experiment. It serves as a perfect example of the clean inheritance pattern, where it sets up its specific loss function and then delegates to the parent's training loop.
*   **Master Engineer's Note:** This class is complete and correct. It should be used as a reference for how to properly structure a child experiment class.

```python
# Assumes all necessary imports are present.

class RlTtmExperiment(MarketExperiment5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_ensemble = []
        self.ttm_loss_fn = None
        self.ttm_weight = 1.0

    def _compute_distillation_loss(self, student_eval: Dict, teacher_eval: Dict, **kwargs) -> torch.Tensor:
        # ... (logic to get student and teacher logits) ...
        ensemble_teacher_logits = torch.stack(teacher_logits_list).mean(dim=0)
        ttm_loss = self.ttm_loss_fn(student_eval['logits'], ensemble_teacher_logits)
        return self.ttm_weight * ttm_loss

    def student_train(self, data, model_params=None, env_params=None, **training_params) -> Dict:
        # --- 1. TTM-specific Setup ---
        pretrained_teacher_paths = training_params.pop('pretrained_teacher_paths', [])
        self.ttm_weight = training_params.get('ttm_weight', 1.0)
        ttm_l = training_params.get('ttm_l', 0.5)
        self.load_teacher_ensemble(pretrained_teacher_paths, model_params)
        self.ttm_loss_fn = TTMLoss(ttm_l=ttm_l).to(self.device)
        
        # --- 2. Delegate to the Parent's Public Training Method ---
        parent_metrics = super().student_train(data, model_params, env_params, **training_params)
        
        # --- 3. Enhance the Standardized Metrics with TTM-specific Info ---
        parent_metrics.update({
            'method': 'TTM',
            'distillation_type': 'task_transfer_matching',
            'ttm_weight': self.ttm_weight,
            'ttm_l': ttm_l,
        })
        return parent_metrics
```

---
### **File 5: The Loss Functions**
*   **Path:** `tsrl/experiments/market/losses.py`
*   **Role:** Contains the implementations of the actual distillation loss functions.
*   **Master Engineer's Note:** This file is a critical dependency. The `_compute_distillation_loss` hooks in the experiment classes will call the functions defined here.

```python
# Assumes all necessary imports are present.

class TTMLoss(nn.Module):
    # ... (full implementation as provided before) ...

def prob_loss(teacher_features, student_features, eps=1e-6, kernel_parameters={}):
    # ... (full implementation of PKT loss as provided before) ...

# ... (and any helper functions like cosine_pairwise_similarities) ...
```

```
Project Path: train_rl_experiments_ttm_pkt.py

Source Tree:

```
train_rl_experiments_ttm_pkt.py

```

`\\?\C:\Users\George\Source\Github\Thesis\financial-trading-in-RL\train_rl_experiments_ttm_pkt.py`:

```py
   1 | """
   2 | TTM vs PKT Comparison Orchestrator - Upgraded Architecture
   3 | 
   4 | This script orchestrates fair comparison experiments between TTM and PKT distillation methods
   5 | by ensuring both methods use identical base parameters and the same pretrained teachers.
   6 | 
   7 | Upgraded to mirror the robust, proven patterns of the original DistillationExperiments orchestrator,
   8 | with comprehensive logging, reporting, and execution flow while maintaining the specialized
   9 | shared teacher pool functionality.
  10 | """
  11 | 
  12 | import yaml
  13 | import numpy as np
  14 | import random
  15 | import torch
  16 | import pandas as pd
  17 | from datetime import datetime
  18 | from pathlib import Path
  19 | from typing import Dict, List, Any, Optional, Tuple
  20 | from uuid import uuid4
  21 | from torch.utils.tensorboard.writer import SummaryWriter
  22 | 
  23 | from experiment_code.data_utils import get_data
  24 | from experiment_code.experiment_utils import (
  25 |     compute_avg_pnl,
  26 |     _calculate_method_aggregates,
  27 |     save_method_data,
  28 |     get_method_index,
  29 |     create_seed_tracker,
  30 |     log_seed_variation_to_tensorboard
  31 | )
  32 | # Legacy reporting imports removed - now using comprehensive academic reporting
  33 | from experiment_code.experiment_checkpointing import (
  34 |     ExperimentCheckpoint,
  35 |     validate_method_experiments
  36 | )
  37 | from experiment_code.metrics import (
  38 |     calculate_max_drawdown,
  39 |     calculate_correlation_matrix,
  40 |     calculate_recovery_time,
  41 |     calculate_annualized_sharpe
  42 | )
  43 | from experiment_code.figure_utils import (
  44 |     add_title,
  45 |     _save_run_visualizations,
  46 |     _plot_pnl_curves
  47 | )
  48 | from experiment_code.tensorboard_utils import (
  49 |     setup_tensorboard,
  50 |     log_metrics,
  51 |     archive_logs,
  52 |     clean_old_logs,
  53 |     create_tensorboard_plots
  54 | )
  55 | from experiment_code.path_validator import PathValidator
  56 | from experiment_code.seed_tracker import SeedTracker
  57 | from experiment_code.academic_integration import AcademicIntegration
  58 | 
  59 | from tsrl.experiments.market.experiment_rl_ttm import RlTtmExperiment
  60 | from tsrl.experiments.market.experiment_offline_pkt import MarketExperiment5
  61 | 
  62 | 
  63 | # Method information mapping for consistent display and reporting
  64 | METHOD_INFO_MAP = {
  65 |     'rl_ttm': {
  66 |         'display_name': 'TTM',
  67 |         'full_name': 'Transformed Teacher Matching',
  68 |         'distillation_type': 'transformed_teacher_matching',
  69 |         'description': 'Reinforcement Learning with Transformed Teacher Matching'
  70 |     },
  71 |     'offline_pkt': {
  72 |         'display_name': 'PKT',
  73 |         'full_name': 'Policy Knowledge Transfer',
  74 |         'distillation_type': 'policy_knowledge_transfer',
  75 |         'description': 'Offline Policy Knowledge Transfer'
  76 |     },
  77 |     'no_distillation': {
  78 |         'display_name': 'Baseline',
  79 |         'full_name': 'No Distillation',
  80 |         'distillation_type': 'none',
  81 |         'description': 'Baseline without distillation'
  82 |     }
  83 | }
  84 | 
  85 | def get_method_info(method_name: str) -> Dict[str, str]:
  86 |     """Get comprehensive method information for display and reporting."""
  87 |     return METHOD_INFO_MAP.get(method_name, {
  88 |         'display_name': method_name,
  89 |         'full_name': method_name,
  90 |         'distillation_type': 'unknown',
  91 |         'description': f'Unknown method: {method_name}'
  92 |     })
  93 | 
  94 | 
  95 | class TTMPKTComparisonOrchestrator:
  96 |     """
  97 |     Orchestrates fair comparison between TTM and PKT methods by mirroring the
  98 |     robust structure of the original DistillationExperiments orchestrator, but
  99 |     with the added capability of managing a shared, pre-trained teacher pool.
 100 |     
 101 |     This ensures both methods use identical teachers for scientific rigor while
 102 |     maintaining all the proven patterns for logging, reporting, and execution.
 103 |     """
 104 |     
 105 |     def __init__(self, config_path: str, use_training: bool = True):
 106 |         with open(config_path, 'r') as f:
 107 |             self.config = yaml.safe_load(f)
 108 |         self.use_training = use_training
 109 | 
 110 |         # --- Adopting the original pathing and ID structure ---
 111 |         self.base_path = Path(self.config['paths']['experiments'])
 112 |         self.figures_path = Path(self.config['paths']['figures'])
 113 |         self.curve_data_path = Path(self.config['paths']['curve_data'])
 114 |         self.config_id = self._create_config_id()
 115 |         
 116 |         for path in [self.base_path, self.figures_path, self.curve_data_path]:
 117 |             path.mkdir(parents=True, exist_ok=True)
 118 |             
 119 |         self._save_config_metadata()
 120 |         self.methods = [config['method'] for config in self.config['distillation']]
 121 |         self.data, self.coin_list_size = get_data(self.config)
 122 |         self.shared_teacher_paths = []
 123 |         
 124 |         # Initialize academic integration for comprehensive evaluation
 125 |         self.academic_integration = AcademicIntegration(self)
 126 |         
 127 |         print(f"✓ TTM vs PKT Orchestrator initialized with STABLE config ID: {self.config_id}")
 128 |         print(f"✓ Academic evaluation system ready for comprehensive reporting")
 129 | 
 130 |     @staticmethod
 131 |     def _get_experiment_class(method_name: str):
 132 |         """Returns appropriate experiment class based on method name."""
 133 |         if method_name == 'no_distillation':
 134 |             # This should point to your baseline experiment class
 135 |             from tsrl.experiments.market.experiment_no_distillation import MarketExperiment1
 136 |             return MarketExperiment1
 137 |         if method_name == 'rl_ttm':
 138 |             return RlTtmExperiment
 139 |         elif method_name == 'offline_pkt':
 140 |             return MarketExperiment5
 141 |         else:
 142 |             raise ValueError(f"This orchestrator only supports 'rl_ttm' and 'offline_pkt' and 'no_distillation', not {method_name}")
 143 | 
 144 |     def _create_config_id(self) -> str:
 145 |         """Creates a human-readable, deterministic ID from key config parameters."""
 146 |         first_method_config = self.config['distillation'][0]
 147 |         training_params = first_method_config['training_params']
 148 |         epochs = training_params.get('n_epochs', 'NA')
 149 |         batch_size = training_params.get('batch_size', 'NA')
 150 |         model_config = self.config.get('model', {})
 151 |         lstm_size = model_config.get('lstm_size', 'NA')
 152 |         return f"e{epochs}_b{batch_size}_l{lstm_size}"
 153 | 
 154 |     def _save_config_metadata(self):
 155 |         """Saves configuration metadata for reference."""
 156 |         metadata_path = self.base_path / self.config_id / 'metadata.yaml'
 157 |         metadata_path.parent.mkdir(exist_ok=True, parents=True)
 158 |         
 159 |         metadata = {
 160 |             'config_id': self.config_id,
 161 |             'timestamp': datetime.now().isoformat(),
 162 |             'config': self.config,
 163 |             'orchestrator_type': 'TTMPKTComparison',
 164 |             'shared_teacher_config': self.config.get('teacher_config', {})
 165 |         }
 166 |         
 167 |         with open(metadata_path, 'w') as f:
 168 |             yaml.dump(metadata, f)
 169 | 
 170 |     def get_checkpoint(self) -> ExperimentCheckpoint:
 171 |         """Gets a checkpoint manager for the current configuration."""
 172 |         return ExperimentCheckpoint(self.base_path, self.config_id)
 173 | 
 174 |     def validate_method_experiments(self, method_name: str, expected_runs: int) -> Dict[str, Any]:
 175 |         """Validates that a method's experiments completed successfully."""
 176 |         return validate_method_experiments(
 177 |             base_path=self.base_path,
 178 |             config_id=self.config_id,
 179 |             method_name=method_name,
 180 |             distillation_names=self.config['paths']['distillation_names'],
 181 |             expected_runs=expected_runs
 182 |         )
 183 | 
 184 |     def run_experiments(self, single_method: Optional[str] = None, resume: bool = False,
 185 |                        generate_thesis_report: bool = True, num_runs: Optional[int] = None) -> Dict:
 186 |         """
 187 |         Main execution loop.
 188 |         MASTER ENGINEER NOTE: This is the corrected version that handles the baseline properly,
 189 |         passes data correctly, and avoids type errors.
 190 |         """
 191 |         if num_runs is None:
 192 |             num_runs = self.config['training'].get('num_runs', 5)
 193 |         
 194 |         print(f"\n=== Starting Comparison Suite ({num_runs} runs per method) ===")
 195 |         
 196 |         checkpoint = self.get_checkpoint()
 197 |         methods_in_config = [mc['method'] for mc in self.config['distillation']]
 198 |         methods_to_run = [single_method] if single_method else methods_in_config
 199 | 
 200 |         # --- Phase 0: Process the Baseline ---
 201 |         avg_baseline_pnl = None
 202 |         baseline_method_name = 'no_distillation'
 203 |         
 204 |         # Check if we need the baseline for comparison plots or if we're running it specifically
 205 |         baseline_is_needed = any(m != baseline_method_name for m in methods_to_run) or baseline_method_name in methods_to_run
 206 | 
 207 |         if baseline_is_needed:
 208 |             print("\n--- Preliminary Step: Loading/Running Baseline for Comparison ---")
 209 |             baseline_config = next((mc for mc in self.config['distillation'] if mc['method'] == baseline_method_name), None)
 210 |             if baseline_config:
 211 |                 try:
 212 |                     baseline_results = self.load_saved_method_experiments(baseline_method_name)
 213 |                     if not baseline_results.get('runs'): raise FileNotFoundError
 214 |                     print(f"✓ Loaded existing baseline results for {baseline_method_name}.")
 215 |                 except (FileNotFoundError, IndexError):
 216 |                     print(f"No valid saved results for {baseline_method_name}. Checking if it needs to be run...")
 217 |                     baseline_results = None
 218 |                 
 219 |                 # Run the baseline only if it's in the list of methods to run and doesn't exist
 220 |                 if baseline_results is None and self.use_training and baseline_method_name in methods_to_run:
 221 |                     print(f"Running baseline experiment: {baseline_method_name}")
 222 |                     baseline_results, _ = self.run_method_experiments(baseline_config, num_runs, is_baseline=True)
 223 |                 
 224 |                 if baseline_results:
 225 |                     pnl_curves = [run['avg_test_pnl'] for run in baseline_results.get('runs', []) if run.get('avg_test_pnl') is not None]
 226 |                     if pnl_curves:
 227 |                         avg_baseline_pnl = pd.concat(pnl_curves, axis=1).mean(axis=1)
 228 |                         print("✓ Average baseline PnL curve is ready for comparison plots.")
 229 |             else:
 230 |                 print(f"⚠️ Warning: '{baseline_method_name}' not found in config. Comparison plots will not have a baseline.")
 231 | 
 232 |         # --- Phase 1: Prepare Shared Teachers (for distillation methods) ---
 233 |         distillation_methods_to_run = [m for m in methods_to_run if m != baseline_method_name]
 234 |         if self.use_training and distillation_methods_to_run:
 235 |             self._prepare_shared_teachers()
 236 |         elif distillation_methods_to_run:
 237 |             self._load_existing_teacher_paths()
 238 | 
 239 |         # --- Phase 2: Run the specified experiments ---
 240 |         all_results = {}
 241 |         for method_config in self.config['distillation']:
 242 |             method_name = method_config['method']
 243 |             
 244 |             # Skip methods not in the execution list
 245 |             if method_name not in methods_to_run:
 246 |                 continue
 247 | 
 248 |             # Skip the baseline here because it's already been processed
 249 |             if method_name == baseline_method_name:
 250 |                 # If we ran the baseline, we need to add its results to the final report
 251 |                 if 'baseline_results' in locals() and baseline_results:
 252 |                     all_results[baseline_method_name] = {'results': baseline_results, 'aggregates': _calculate_method_aggregates(baseline_results)}
 253 |                 continue
 254 |                 
 255 |             print(f"\n=== Processing method: {method_name} ===")
 256 |             
 257 |             method_results, agg_results = self.run_method_experiments(
 258 |                 method_config=method_config,
 259 |                 num_runs=num_runs,
 260 |                 is_baseline=False,
 261 |                 avg_baseline_pnl=avg_baseline_pnl # <-- Pass the baseline PnL
 262 |             )
 263 |             
 264 |             all_results[method_name] = {'results': method_results, 'aggregates': agg_results}
 265 |             
 266 |             # Validate and mark completed
 267 |             validation = self.validate_method_experiments(method_name, num_runs)
 268 |             if validation['success']:
 269 |                 checkpoint.method_completed(method_name, validation)
 270 |                 print(f"✅ Method {method_name} marked as completed")
 271 |             else:
 272 |                 print(f"⚠️ Validation for {method_name} failed: {validation['reason']}")
 273 |             
 274 |         # Generate comprehensive academic evaluation reports
 275 |         print("\nGenerating comprehensive academic evaluation reports...")
 276 |         try:
 277 |             academic_files = self.academic_integration.generate_academic_report(all_results)
 278 |             print(f"✓ Academic reports generated: {len(academic_files)} files")
 279 |             for file_path in academic_files:
 280 |                 print(f"  - {file_path}")
 281 |         except Exception as e:
 282 |             print(f"⚠️ Academic report generation failed: {e}")
 283 |             # Continue execution - don't fail the entire experiment for reporting issues
 284 | 
 285 |         # Final validation
 286 |         validator = PathValidator(self.config_id, Path.cwd())
 287 |         current_mode = 'train' if self.use_training else 'evaluate'
 288 |         validation_results = validator.validate_all(self.methods, mode=current_mode)
 289 |         if validation_results:
 290 |             print("\nDirectory structure validation found issues:")
 291 |             validator.print_validation_report(validation_results)
 292 |         else:
 293 |             print("\nDirectory structure validated successfully")
 294 |             
 295 |         print("\n=== TTM vs PKT Comparison Suite Complete ===")
 296 |         return all_results
 297 | 
 298 |     def run_method_experiments(self, method_config: Dict, num_runs: int, is_baseline: bool = False, avg_baseline_pnl: Optional[pd.Series] = None) -> Tuple[Dict, Dict]:
 299 |         """
 300 |         Runs multiple experiments for a single method.
 301 |         MASTER ENGINEER NOTE: Added is_baseline flag.
 302 |         """
 303 |         method_name = method_config['method']
 304 |         print(f"\n--- Running {num_runs} experiments for {method_name} ---")
 305 |         
 306 |         method_idx = get_method_index(method_name)
 307 |         method_path = self.base_path / self.config_id / self.config['paths']['distillation_names'][method_idx]
 308 |         method_path.mkdir(parents=True, exist_ok=True)
 309 | 
 310 |         results = {
 311 |             'pnl_curves': [],
 312 |             'training_metrics': [],
 313 |             'positions': [],
 314 |             'sharpe_ratios': [],
 315 |             'final_metrics': [],
 316 |             'runs': [],
 317 |             'config_id': self.config_id,
 318 |             'method': method_name
 319 |         }
 320 |         
 321 |         # Create seed tracker for this method
 322 |         seed_tracker = create_seed_tracker(
 323 |             method_name=method_name,
 324 |             config_id=self.config_id,
 325 |             num_seeds=num_runs
 326 |         )
 327 | 
 328 |         # Setup TensorBoard writer for this method
 329 |         writer = setup_tensorboard(self.config_id, method_name, clean=True)
 330 | 
 331 |         for run_idx in range(num_runs):
 332 |             try:
 333 |                 # Mirror the original distillation orchestrator pattern - no seed_tracker.get_seed()
 334 |                 print(f"\n--- {method_name} Run {run_idx + 1}/{num_runs} ---")
 335 |                 
 336 |                 timestamp = datetime.now().strftime('%Y%m%d-%H%M-')
 337 |                 random_value = str(uuid4())[:4]
 338 |                 unique_id = f"{timestamp}{random_value}"
 339 |                 
 340 |                 exp_path = method_path / f"{method_name}_experiment_{unique_id}"
 341 |                 exp_path.mkdir(parents=True, exist_ok=True)
 342 |                 
 343 |                 models_dir = exp_path / "models"
 344 |                 models_dir.mkdir(parents=True, exist_ok=True)
 345 |                 
 346 |                 # --- THE KEY UNIFICATION ---
 347 |                 training_params = method_config['training_params'].copy()
 348 |                 if not is_baseline:
 349 |                     # Inject shared teacher paths only for distillation methods
 350 |                     training_params['pretrained_teacher_paths'] = self.shared_teacher_paths
 351 |                 
 352 |                 run_results = self._execute_single_run(
 353 |                     method_name=method_name,
 354 |                     method_config=method_config,
 355 |                     training_params=training_params,
 356 |                     exp_path=exp_path,
 357 |                     run_idx=run_idx,
 358 |                     models_dir=models_dir,
 359 |                     unique_id=unique_id,
 360 |                     writer=writer
 361 |                 )
 362 |                 
 363 |                 # Calculate additional metrics
 364 |                 sharpe = calculate_annualized_sharpe(run_results['avg_test_pnl'])
 365 |                 run_results['sharpe_ratio'] = sharpe
 366 |                 results['sharpe_ratios'].append(sharpe)
 367 |                 
 368 |                 results['pnl_curves'].append(run_results['avg_test_pnl'])
 369 |                 results['training_metrics'].append(run_results['training_metrics'])
 370 |                 run_results['unique_id'] = unique_id
 371 |                 run_results['timestamp'] = timestamp
 372 |                 results['runs'].append(run_results)
 373 |                 
 374 |                 # Save run visualizations
 375 |                 _save_run_visualizations(
 376 |                     data=self.data,
 377 |                     figures_path=self.figures_path,
 378 |                     method_name=method_name,
 379 |                     run_idx=run_idx,
 380 |                     run_results=run_results,
 381 |                     fig_avg_pnl=None,  # Will be created inside the function
 382 |                     unique_id=unique_id,
 383 |                     config_id=self.config_id,
 384 |                     avg_baseline_pnl=avg_baseline_pnl,
 385 |                     method_display_names={method_name: method_config.get('display_name', method_name)}
 386 |                 )
 387 |                 
 388 |                 # Track run in seed tracker
 389 |                 seed_tracker.track_run(run_idx, run_results)
 390 |                 
 391 |                 # Log metrics to TensorBoard
 392 |                 summary_metrics = {
 393 |                     'PnL': run_results['avg_test_pnl'].iloc[-1],
 394 |                     'Sharpe': sharpe,
 395 |                     'Max_Drawdown': calculate_max_drawdown(run_results['avg_test_pnl'])
 396 |                 }
 397 |                 
 398 |                 # Add training metrics if available
 399 |                 if run_results['training_metrics']:
 400 |                     if isinstance(run_results['training_metrics'], dict):
 401 |                         for k, v in run_results['training_metrics'].items():
 402 |                             if isinstance(v, (list, np.ndarray)) and len(v) > 0:
 403 |                                 summary_metrics[f"train_{k}_final"] = v[-1]
 404 |                             elif isinstance(v, (int, float)):
 405 |                                 summary_metrics[f"train_{k}"] = v
 406 |                 
 407 |                 log_metrics(writer, "run_summary", summary_metrics, run_idx)
 408 |                 
 409 |             except Exception as e:
 410 |                 print(f"Error in run {run_idx}: {str(e)}")
 411 |                 import traceback
 412 |                 traceback.print_exc()
 413 |                 continue
 414 | 
 415 |         # Log seed variation to TensorBoard
 416 |         seed_tracker.log_to_tensorboard(writer)
 417 |         
 418 |         # Create additional TensorBoard visualizations
 419 |         create_tensorboard_plots(writer, method_name, results, self.config_id)
 420 |         
 421 |         # Calculate method aggregates
 422 |         agg_results = _calculate_method_aggregates(results)
 423 |         
 424 |         # Save results
 425 |         save_method_data(self.base_path, self.curve_data_path, self.config,
 426 |                         method_name, results, agg_results, self.config_id)
 427 |         
 428 |         # Close TensorBoard writer
 429 |         writer.close()
 430 |         
 431 |         return results, agg_results
 432 | 
 433 |     def _execute_single_run(self, method_name: str, method_config: Dict, training_params: Dict,
 434 |                             exp_path: Path, run_idx: int, models_dir: Path, unique_id: str,
 435 |                             writer: Optional[SummaryWriter] = None) -> Dict:
 436 |         """
 437 |         Executes a single run.
 438 |         MASTER ENGINEER NOTE: Now handles the different train method signatures.
 439 |         """
 440 |         exp_class = self._get_experiment_class(method_name)
 441 |         model_params = self.config['model'].copy()
 442 |         model_params['num_inputs'] = list(self.data['feature_dict'].values())[0].shape[1]
 443 |         
 444 |         experiment = exp_class(exp_path=exp_path, models_dir=models_dir, unique_id=unique_id)
 445 | 
 446 |         training_metrics = {}
 447 |         if self.use_training:
 448 |             # --- TACTICAL DISPATCH BASED ON METHOD NAME ---
 449 |             if method_name == 'no_distillation':
 450 |                 # Call the baseline's unique 'train' method
 451 |                 training_metrics = experiment.train(
 452 |                     data=self.data,
 453 |                     model_params=model_params,
 454 |                     env_params=self.config['training']['env_params'],
 455 |                     seed=run_idx,
 456 |                     **training_params
 457 |                 )
 458 |             else:
 459 |                 # Call the standard 'student_train' for all distillation methods
 460 |                 training_metrics = experiment.student_train(
 461 |                     data=self.data,
 462 |                     model_params=model_params,
 463 |                     env_params=self.config['training']['env_params'],
 464 |                     seed=run_idx,
 465 |                     **training_params
 466 |                 )
 467 |         print(f"  > [Step 2/3] Evaluating final model...")
 468 |         eval_dict = experiment.eval(self.data)
 469 |         print(f"  > Evaluation complete.")
 470 | 
 471 |         # === STEP 3: BACKTEST AND CALCULATE PERFORMANCE METRICS ===
 472 |         print(f"  > [Step 3/3] Backtesting and calculating performance...")
 473 |         detailed_pnls = experiment.detailed_backtest(
 474 |             self.data, eval_dict, train_end=self.config['data']['train_end']
 475 |         )
 476 |         final_performance_report = experiment.backtest(
 477 |             self.data, eval_dict, train_end=self.config['data']['train_end']
 478 |         )
 479 |         print(f"  > Final Independent Test PnL: {final_performance_report.get('test_pnl', 0.0):.4f}")
 480 |         _, _, avg_only_test_pnl = compute_avg_pnl(
 481 |             detailed_pnls, self.coin_list_size
 482 |     )
 483 |         return {
 484 |         'pnl_curves': detailed_pnls,
 485 |         'eval_dict': eval_dict,
 486 |         'training_metrics': training_metrics,
 487 |         'final_performance_metrics': final_performance_report,
 488 |         'avg_test_pnl': avg_only_test_pnl,
 489 |         'unique_id': unique_id
 490 |     }
 491 | 
 492 |     def _prepare_shared_teachers(self):
 493 |         """Prepares the shared teacher pool using the robust teacher management system."""
 494 |         teacher_config = self.config.get('teacher_config', {})
 495 |         teacher_path = self.base_path / self.config_id / teacher_config.get('teacher_save_dir', 'shared_teachers')
 496 |         teacher_path.mkdir(parents=True, exist_ok=True)
 497 |         
 498 |         n_teachers_to_select = teacher_config.get('n_teachers_to_select', 5)
 499 |         teacher_pool_size = teacher_config.get('teacher_pool_size', 8)
 500 |         
 501 |         # Check if we already have sufficient teachers
 502 |         existing_teachers = list(teacher_path.glob("teacher_*.pt"))
 503 |         if len(existing_teachers) >= n_teachers_to_select:
 504 |             print(f"✓ Found {len(existing_teachers)} existing teachers. Using the top {n_teachers_to_select}.")
 505 |             self.shared_teacher_paths = [str(p) for p in existing_teachers[:n_teachers_to_select]]
 506 |             
 507 |             # Save teacher metadata for reference
 508 |             self._save_teacher_metadata(teacher_path, existing_teachers[:n_teachers_to_select])
 509 |             return
 510 | 
 511 |         print(f"--- Preparing Shared Teachers ({teacher_pool_size} models) ---")
 512 |         
 513 |         # Use MarketExperiment5 as a utility for training teachers
 514 |         teacher_trainer = MarketExperiment5(
 515 |             exp_path=teacher_path / "trainer_temp",
 516 |             use_sentiment=False,
 517 |             models_dir=str(teacher_path)
 518 |         )
 519 |         teacher_trainer.db['env_params'] = self.config['training']['env_params']
 520 |         
 521 |         # Get teacher training params from config
 522 |         teacher_training_params = self._get_teacher_training_params(teacher_config)
 523 |         
 524 |         try:
 525 |             all_teacher_paths = teacher_trainer.train_and_save_ensemble(
 526 |                 data=self.data,
 527 |                 output_dir=teacher_path,
 528 |                 teacher_model_params=self.config['model'],
 529 |                 env_params=self.config['training']['env_params'],
 530 |                 n_teachers=teacher_pool_size,
 531 |                 seed=teacher_config.get('base_seed', 42),
 532 |                 **teacher_training_params
 533 |             )
 534 |             
 535 |             print("\n--- Selecting Best Teachers ---")
 536 |             teacher_ensemble = teacher_trainer.load_teacher_ensemble(all_teacher_paths, self.config['model'])
 537 |             selected_names = teacher_trainer.select_best_teachers(
 538 |                 data=self.data,
 539 |                 teacher_ensemble=teacher_ensemble,
 540 |                 eval_config={'train_end': self.config['data']['train_end']},
 541 |                 n_select=n_teachers_to_select
 542 |             )
 543 |             
 544 |             # Filter the paths to only the selected teachers
 545 |             self.shared_teacher_paths = [p for p in all_teacher_paths if any(name in str(p) for name in selected_names)]
 546 |             print(f"✓ Prepared and selected {len(self.shared_teacher_paths)} elite teachers.")
 547 |             
 548 |             # Save comprehensive teacher metadata
 549 |             self._save_teacher_metadata(teacher_path, self.shared_teacher_paths, {
 550 |                 'teacher_pool_size': teacher_pool_size,
 551 |                 'n_teachers_selected': len(self.shared_teacher_paths),
 552 |                 'base_seed': teacher_config.get('base_seed', 42),
 553 |                 'training_params': teacher_training_params,
 554 |                 'selected_names': selected_names,
 555 |                 'selection_criteria': 'best_performance_on_validation'
 556 |             })
 557 |             
 558 |         except Exception as e:
 559 |             print(f"✗ Teacher preparation failed: {str(e)}")
 560 |             import traceback
 561 |             traceback.print_exc()
 562 |             self.shared_teacher_paths = []
 563 | 
 564 |     def _load_existing_teacher_paths(self):
 565 |         """Loads paths to existing teachers when not training."""
 566 |         teacher_config = self.config.get('teacher_config', {})
 567 |         teacher_path = self.base_path / self.config_id / teacher_config.get('teacher_save_dir', 'shared_teachers')
 568 |         n_teachers_to_select = teacher_config.get('n_teachers_to_select', 5)
 569 |         
 570 |         existing_teachers = list(teacher_path.glob("teacher_*.pt"))
 571 |         if not existing_teachers:
 572 |             print(f"⚠️ Warning: No teachers found in {teacher_path}")
 573 |             self.shared_teacher_paths = []
 574 |             return
 575 |             
 576 |         self.shared_teacher_paths = [str(p) for p in existing_teachers[:n_teachers_to_select]]
 577 |         print(f"✓ Loaded {len(self.shared_teacher_paths)} existing teachers for evaluation.")
 578 | 
 579 |     def _get_teacher_training_params(self, teacher_config: Dict) -> Dict:
 580 |         """Get comprehensive teacher training parameters from config."""
 581 |         base_params = {
 582 |             'n_epochs': teacher_config.get('teacher_epochs', 100),
 583 |             'batch_size': 32,
 584 |             'learning_rate': 5e-4,
 585 |             'ppo_clip': 0.2,
 586 |             'n_envs': 128,
 587 |             'n_reuse_value': 1,
 588 |             'use_amp': False,
 589 |             'rew_limit': 6.0,
 590 |             'truncate_bptt': (5, 20),
 591 |             'tau': 0.95,
 592 |             'env_step_init': 1.0,
 593 |             'validation_interval': 502,
 594 |             'show_progress': True,
 595 |             'weight_decay': 0.0,
 596 |             'entropy_weight': 0.01,
 597 |             'recompute_values': False,
 598 |             'value_horizon': float('inf'),
 599 |             'lookahead': False,
 600 |             'advantage_type': 'direct_reward',
 601 |             'gamma': 0.99,
 602 |             'n_reuse_policy': 3,
 603 |             'n_reuse_aux': 0,
 604 |             'checkpoint_dir': False,
 605 |         }
 606 |         
 607 |         # Override with any teacher-specific params from config
 608 |         teacher_training_params = teacher_config.get('teacher_training_params', {})
 609 |         base_params.update(teacher_training_params)
 610 |         
 611 |         return base_params
 612 | 
 613 |     def _save_teacher_metadata(self, teacher_path: Path, teacher_paths: List[str], extra_info: Dict = None):
 614 |         """Save comprehensive teacher metadata for reproducibility and reuse."""
 615 |         teacher_metadata = {
 616 |             'teacher_paths': [str(p) for p in teacher_paths],
 617 |             'n_teachers': len(teacher_paths),
 618 |             'timestamp': datetime.now().isoformat(),
 619 |             'config_id': self.config_id,
 620 |             'teacher_config': self.config.get('teacher_config', {}),
 621 |             'model_params': self.config['model']
 622 |         }
 623 |         
 624 |         if extra_info:
 625 |             teacher_metadata.update(extra_info)
 626 |         
 627 |         metadata_path = teacher_path / "shared_teachers_metadata.yaml"
 628 |         with open(metadata_path, 'w') as f:
 629 |             yaml.dump(teacher_metadata, f, default_flow_style=False)
 630 |         
 631 |         print(f"✓ Teacher metadata saved to: {metadata_path}")
 632 | 
 633 |     def generate_thesis_report(self, results: Dict):
 634 |         """Generate thesis-style report with all Chapter 4 visualizations."""
 635 |         print("\n=== Generating Thesis Chapter 4 Report ===")
 636 |         
 637 |         # Create properly formatted data structure for our report generator
 638 |         experiment_results = {}
 639 |         
 640 |         for method_name, method_data in results.items():
 641 |             if 'results' not in method_data or 'aggregates' not in method_data:
 642 |                 print(f"Warning: Skipping method {method_name} in thesis report due to missing keys.")
 643 |                 continue
 644 |             
 645 |             method_results = method_data['results']
 646 |             agg_data = method_data['aggregates']
 647 |             
 648 |             # Ensure performance metrics exist
 649 |             if 'performance' not in agg_data or 'cumulative_pnl_mean' not in agg_data['performance']:
 650 |                 print(f"Warning: Skipping method {method_name} in thesis report due to missing performance metrics.")
 651 |                 continue
 652 |             
 653 |             experiment_results[method_name] = {
 654 |                 'runs': method_results.get('runs', []),
 655 |                 'final_metrics': {
 656 |                     'pnl': {
 657 |                         'mean': agg_data['performance']['cumulative_pnl_mean'],
 658 |                         'std': agg_data['performance']['cumulative_pnl_std']
 659 |                     }
 660 |                 },
 661 |                 'aggregates': agg_data
 662 |             }
 663 |         
 664 |         # Create academic report using unified reporting pipeline
 665 |         output_dir = self.base_path / self.config_id / "thesis_report"
 666 |         print(f"Generating academic report in {output_dir}")
 667 |         
 668 |         try:
 669 |             academic_integration = AcademicIntegration(str(output_dir))
 670 |             output_files = academic_integration.generate_academic_report(experiment_results)
 671 |             
 672 |             print("Thesis report generated successfully!")
 673 |             print(f"Report location: {output_dir}")
 674 |             print("Generated files:")
 675 |             for name, path in output_files.items():
 676 |                 if hasattr(path, 'relative_to'):
 677 |                     try:
 678 |                         rel_path = path.relative_to(Path.cwd())
 679 |                     except ValueError:
 680 |                         rel_path = path
 681 |                     print(f"- {name}: {rel_path}")
 682 |                 else:
 683 |                     print(f"- {name}")
 684 |         except Exception as e:
 685 |             print(f"Error generating thesis report: {str(e)}")
 686 |             import traceback
 687 |             traceback.print_exc()
 688 | 
 689 |     def load_saved_method_experiments(self, method_name: str) -> Dict:
 690 |         """Loads all saved experiments for a particular method."""
 691 |         method_path = (self.base_path /
 692 |                       self.config_id /
 693 |                       self.config['paths']['distillation_names'][
 694 |                           get_method_index(method_name)
 695 |                       ])
 696 | 
 697 |         results = {
 698 |             'runs': [],
 699 |             'pnl_curves': [],
 700 |             'config_id': self.config_id
 701 |         }
 702 | 
 703 |         experiment_paths = list(method_path.glob(f"{method_name}_experiment_*"))
 704 | 
 705 |         if not experiment_paths:
 706 |             print(f"Warning: No saved experiments found for {method_name} in {method_path}")
 707 |             return results
 708 | 
 709 |         print(f"\nLoading {len(experiment_paths)} saved experiments for {method_name}")
 710 | 
 711 |         for idx, exp_path in enumerate(experiment_paths):
 712 |             try:
 713 |                 run_results = self._load_saved_experiment(method_name, exp_path)
 714 |                 results['runs'].append(run_results)
 715 |                 results['pnl_curves'].append(run_results['avg_test_pnl'])
 716 |                 print(f"Successfully loaded experiment from {exp_path.name}")
 717 |             except Exception as e:
 718 |                 print(f"Error loading experiment from {exp_path}: {str(e)}")
 719 |                 continue
 720 | 
 721 |         return results
 722 | 
 723 |     def _load_saved_experiment(self, method_name: str, experiment_path: Path) -> Dict:
 724 |         """Loads a saved experiment and evaluates it."""
 725 |         exp_class = self._get_experiment_class(method_name)
 726 |         experiment = exp_class(
 727 |             exp_path=experiment_path,
 728 |             use_sentiment=False,
 729 |             models_dir=experiment_path / "models",
 730 |             unique_id=experiment_path.name.split('_')[-1]
 731 |         )
 732 | 
 733 |         state_dict_path = experiment_path / "exp_state_dict.pkl"
 734 |         if not state_dict_path.exists():
 735 |             raise FileNotFoundError(f"State dict not found at {state_dict_path}")
 736 | 
 737 |         print(f"\nEvaluating saved experiment: {experiment_path.name}")
 738 |         eval_dict = experiment.eval(self.data, dir=str(state_dict_path))
 739 |         detailed_pnls = experiment.detailed_backtest(
 740 |             self.data,
 741 |             eval_dict,
 742 |             train_end=self.config['data']['train_end']
 743 |         )
 744 | 
 745 |         avg_train_pnl, avg_test_pnl, avg_only_test_pnl = compute_avg_pnl(
 746 |             detailed_pnls=detailed_pnls,
 747 |             coin_list_size=self.coin_list_size
 748 |         )
 749 | 
 750 |         return {
 751 |             'pnl_curves': detailed_pnls,
 752 |             'eval_dict': eval_dict,
 753 |             'avg_test_pnl': avg_only_test_pnl,
 754 |             'timestamp': experiment_path.name.split('_')[2],
 755 |             'unique_id': experiment_path.name.split('_')[-1],
 756 |             'config_id': self.config_id
 757 |         }
 758 | 
 759 | 
 760 | def main():
 761 |     """Main entry point for TTM vs PKT comparison experiments."""
 762 |     import argparse
 763 |     
 764 |     parser = argparse.ArgumentParser(description='Run TTM vs PKT comparison experiments')
 765 |     parser.add_argument('mode', choices=['train', 'evaluate'],
 766 |                        help='Mode to run: train or evaluate models')
 767 |     parser.add_argument('--config', default='experiment_config_ttm_pkt.yaml',
 768 |                        help='Path to experiment configuration file')
 769 |     parser.add_argument('--resume', action='store_true',
 770 |                        help='Resume from checkpoint (skip completed methods)')
 771 |     parser.add_argument('--method', type=str, default=None,
 772 |                        help='Run only a specific method (e.g., "rl_ttm")')
 773 |     parser.add_argument('--runs', type=int, default=None,
 774 |                        help='Number of runs per method (overrides config)')
 775 |     parser.add_argument('--no-thesis-report', action='store_true',
 776 |                        help='Disable thesis-style report generation')
 777 |     
 778 |     args = parser.parse_args()
 779 |     
 780 |     try:
 781 |         # Initialize orchestrator
 782 |         orchestrator = TTMPKTComparisonOrchestrator(
 783 |             config_path=args.config,
 784 |             use_training=(args.mode == 'train')
 785 |         )
 786 |         
 787 |         # Run experiments
 788 |         results = orchestrator.run_experiments(
 789 |             single_method=args.method,
 790 |             resume=args.resume,
 791 |             generate_thesis_report=not args.no_thesis_report,
 792 |             num_runs=args.runs
 793 |         )
 794 |         
 795 |         print("\n=== Comparison Summary ===")
 796 |         for method_name, method_data in results.items():
 797 |             successful = len([r for r in method_data['results']['runs'] if 'error' not in r])
 798 |             total = len(method_data['results']['runs'])
 799 |             print(f"{method_name}: {successful}/{total} successful runs")
 800 |         
 801 |         print(f"\nResults and reports saved to: {orchestrator.base_path / orchestrator.config_id}")
 802 |         
 803 |     except KeyboardInterrupt:
 804 |         print("\n✗ Experiment interrupted by user")
 805 |     except Exception as e:
 806 |         print(f"\n✗ Experiment failed: {str(e)}")
 807 |         import traceback
 808 |         traceback.print_exc()
 809 |         raise
 810 | 
 811 | 
 812 | if __name__ == "__main__":
 813 |     main()

```
```



Project Path: experiment_rl_ttm.py

Source Tree:

```
experiment_rl_ttm.py

```

`\\?\C:\Users\George\Source\Github\Thesis\financial-trading-in-RL\tsrl\experiments\market\experiment_rl_ttm.py`:

```py
   1 | import torch
   2 | from pathlib import Path
   3 | from typing import Dict, List, Any
   4 | 
   5 | from tsrl.experiments.market.experiment_offline_pkt import MarketExperiment5
   6 | from tsrl.experiments.market.model import MarketAgent
   7 | from tsrl.experiments.market.losses import TTMLoss
   8 | 
   9 | class RlTtmExperiment(MarketExperiment5):
  10 |     """
  11 |     Implements Transformed Teacher Matching (TTM) for RL by correctly
  12 |     overriding the distillation loss calculation from MarketExperiment5.
  13 |     
  14 |     This class is now architecturally sound, containing ONLY the logic
  15 |     that is unique to TTM.
  16 |     """
  17 | 
  18 |     def __init__(self, *args, **kwargs):
  19 |         super().__init__(*args, **kwargs)
  20 |         self.teacher_ensemble = []
  21 |         self.ttm_loss_fn = None
  22 |         self.ttm_weight = 1.0
  23 | 
  24 |     def _initialize_ttm_ensemble(self, model_params: Dict[str, Any], pretrained_paths: List[str]):
  25 |         """Initializes the TTM teacher ensemble and stores it as an instance variable."""
  26 |         if not pretrained_paths:
  27 |             raise ValueError("TTM requires 'pretrained_teacher_paths'. None were provided.")
  28 | 
  29 |         ensemble = []
  30 |         print(f"\nInitializing TTM Teacher Ensemble ({len(pretrained_paths)} teachers)...")
  31 | 
  32 |         for i, teacher_path in enumerate(pretrained_paths):
  33 |             try:
  34 |                 teacher_model = MarketAgent(**model_params)
  35 |                 checkpoint = torch.load(teacher_path, map_location=self.device)
  36 |                 state_dict = checkpoint.get('model_state_dict', checkpoint)
  37 |                 teacher_model.load_state_dict(state_dict)
  38 |                 teacher_model.to(self.device)
  39 |                 teacher_model.eval()
  40 |                 ensemble.append(teacher_model)
  41 |                 print(f"  ✓ Loaded teacher {i+1} from {Path(teacher_path).name}")
  42 |             except Exception as e:
  43 |                 raise IOError(f"Could not load teacher from {teacher_path}") from e
  44 |         
  45 |         self.teacher_ensemble = ensemble
  46 | 
  47 |     def _compute_distillation_loss(self, student_eval: Dict, teacher_eval: Dict, **kwargs) -> torch.Tensor:
  48 |         """
  49 |         OVERRIDDEN HOOK: This is the core of the TTM implementation.
  50 |         It replaces the parent's PKT loss calculation with the TTM ensemble loss.
  51 |         
  52 |         The `teacher_eval` from the parent is ignored here, as we use our own ensemble.
  53 |         """
  54 |         # The 'features', 'position', 'actions' needed for teacher predictions must be passed via kwargs
  55 |         features = kwargs.get("features")
  56 |         position = kwargs.get("position")
  57 |         actions = kwargs.get("actions")
  58 | 
  59 |         if features is None or position is None or actions is None:
  60 |             raise ValueError("_compute_distillation_loss for TTM requires 'features', 'position', and 'actions' kwargs.")
  61 | 
  62 |         # 1. Gather predictions from all teachers in the ensemble
  63 |         teacher_logits_list = []
  64 |         with torch.no_grad():
  65 |             for teacher in self.teacher_ensemble:
  66 |                 # Note: We assume teachers use the same input format
  67 |                 t_eval = teacher.train_eval(actions=actions, features=features, position=position)
  68 |                 teacher_logits_list.append(t_eval['logits'])
  69 |         
  70 |         # 2. Average the teacher logits to create a single ensemble prediction
  71 |         ensemble_teacher_logits = torch.stack(teacher_logits_list).mean(dim=0)
  72 | 
  73 |         # 3. Compute the TTM loss against the student's logits
  74 |         ttm_loss = self.ttm_loss_fn(student_eval['logits'], ensemble_teacher_logits)
  75 |         
  76 |         return self.ttm_weight * ttm_loss
  77 | 
  78 | 
  79 |     def student_train(self, data, model_params=None, env_params=None, **training_params) -> Dict:
  80 |         """
  81 |         Overrides the parent's student_train to perform TTM-specific setup,
  82 |         delegate the core training loop, and enhance the returned metrics.
  83 |         """
  84 |         # --- 1. TTM-specific Setup ---
  85 |         # This part handles the logic unique to the TTM experiment.
  86 |         pretrained_teacher_paths = training_params.get('pretrained_teacher_paths', []) # Use .get for safety
  87 |         if not pretrained_teacher_paths:
  88 |             raise ValueError("TTM experiment requires 'pretrained_teacher_paths'.")
  89 | 
  90 |         self.ttm_weight = training_params.get('ttm_weight', 1.0)
  91 |         ttm_l = training_params.get('ttm_l', 0.5)
  92 | 
  93 |         self._initialize_ttm_ensemble(model_params, pretrained_teacher_paths)
  94 |         self.ttm_loss_fn = TTMLoss(ttm_l=ttm_l).to(self.device)
  95 |         
  96 |         print(f"\nTTM setup complete. Delegating to parent PPO training loop.")
  97 |         
  98 |         # --- 2. Delegate to the Parent's Public Training Method ---
  99 |         # The `super().student_train()` call now invokes the clean adapter method
 100 |         # we created in Phase 1, which returns a standardized dictionary.
 101 |         parent_training_metrics = super().student_train(data, model_params, env_params, **training_params)
 102 |         
 103 |         # --- 3. Enhance the Standardized Metrics with TTM-specific Info ---
 104 |         # We now add/overwrite keys to provide context for this specific run.
 105 |         parent_training_metrics.update({
 106 |             'method': 'TTM',
 107 |             'distillation_type': 'task_transfer_matching',
 108 |             'ttm_weight': self.ttm_weight,
 109 |             'ttm_l': ttm_l,
 110 |         })
 111 | 
 112 |         return parent_training_metrics

```Project Path: experiment_offline_pkt.py

Source Tree:

```
experiment_offline_pkt.py

```

`\\?\C:\Users\George\Source\Github\Thesis\financial-trading-in-RL\tsrl\experiments\market\experiment_offline_pkt.py`:

```py
   1 | from tsrl.utils.torch_base import TorchExperiment
   2 | from tsrl.torch_utils import to_np
   3 | from tsrl.torch_utils.optim import Lookahead, RAdam
   4 | from torch.optim import Adam
   5 | from tsrl.utils import create_decay, create_hyperbolic_decay, random_batched_iterator, RunningMeanStd, RunningScalarMean
   6 | from tsrl.environments.market_env import VecCandleMarketEnv
   7 | from tsrl.environments.wrappers import PytorchConverter, NormalizeWrapper
   8 | from tsrl.environments import generate_candle_features, create_pair_sample_ranges
   9 | from tsrl.advantage import NormalizedGAE, calculate_gae_hyperbolic, calculate_advantage_vectorized
  10 | from tsrl.algorithms.ppo import ppo_categorical_policy_loss, ppo_value_loss
  11 | from tsrl.experiments.market.model import MarketAgent
  12 | import random
  13 | from tsrl.environments import generate_candle_features
  14 | 
  15 | 
  16 | from fin_utils.pnl import pnl_from_price_position
  17 | 
  18 | import torch
  19 | from torch import nn
  20 | import torch.nn.functional as F
  21 | from torch.utils.tensorboard import SummaryWriter
  22 | from torch.distributions import kl_divergence, Categorical
  23 | from ray import tune
  24 | 
  25 | import numpy as np
  26 | import pandas as pd
  27 | from tqdm import tqdm
  28 | 
  29 | from pathlib import Path
  30 | from typing import Dict, Tuple, Optional, Union
  31 | from collections import defaultdict, deque
  32 | 
  33 | import matplotlib.pyplot as plt
  34 | import seaborn as sns
  35 | 
  36 | from plotly import graph_objects as go
  37 | import os
  38 | import warnings
  39 | import argparse
  40 | from torch.autograd import Variable
  41 | parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
  42 | warnings.filterwarnings('ignore')
  43 | os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  44 | 
  45 | 
  46 | 
  47 | @torch.no_grad()
  48 | def gather_trajectories(env, model: MarketAgent, include_value=False, keep_states=False):
  49 |     """
  50 | 
  51 |     :param env: Environment to step through
  52 |     :param model: Model to use for inference through environment
  53 |     :param include_value: whether to include the value function estimation in the trajectory data
  54 |     :param keep_states: whether to keep the hidden state return by the model in the trajectory data (e.g. LSTM state)
  55 |     :return:
  56 |     """
  57 |     trajectories = defaultdict(list)
  58 |     rews, obs, is_first = env.observe()
  59 |     trajectories['obs'].append(obs)
  60 |     state = None
  61 |     while True:
  62 |         pred = model.sample_eval(state=state, **obs)
  63 |         if keep_states:
  64 |             trajectories['states'].append(state)
  65 |         state = pred['state']
  66 |         env.act(dict(market_action=pred['action']))
  67 |         rews, obs, is_first = env.observe()
  68 |         trajectories['action'].append(pred['action'].detach())
  69 |         trajectories['log_prob'].append(pred['log_prob'].detach())
  70 |         trajectories['logits'].append(pred['logits'].detach())
  71 |         if include_value:
  72 |             trajectories['value'].append(pred['value'].detach().to('cpu', non_blocking=True))
  73 |         trajectories['rews'].append(rews)
  74 |         if any(is_first):
  75 |             assert all(is_first)
  76 |             break
  77 |         trajectories['obs'].append(obs)
  78 |     return trajectories
  79 | 
  80 | 
  81 | def prepare_trajectory_windows(trajectories):
  82 |     # obs shapes : step x n_envs x nb_features : transpose(0,1)
  83 |     rew_array = np.asarray(trajectories['rews']).T
  84 |     obs_tensor_dict = dict()
  85 |     for k in trajectories['obs'][0].keys():
  86 |         obs_tensor_dict[k] = torch.stack([tob[k] for tob in trajectories['obs']], dim=1).detach()
  87 |     old_log_prob_tensor = torch.stack(trajectories['log_prob'], dim=1).detach()
  88 |     assert old_log_prob_tensor.shape[-1] == 1
  89 |     old_log_prob_tensor = old_log_prob_tensor[..., 0]
  90 |     action_tensor = torch.cat(trajectories['action'], dim=1).detach()
  91 |     logits = torch.cat(trajectories['logits'], dim=1).detach()
  92 |     if 'value' in trajectories:
  93 |         value_tensor = to_np(torch.stack(trajectories['value'], dim=1))
  94 |         return rew_array, obs_tensor_dict, old_log_prob_tensor, action_tensor, logits, value_tensor
  95 |     return rew_array, obs_tensor_dict, old_log_prob_tensor, action_tensor, logits
  96 | 
  97 | 
  98 | """
  99 | Class for deep reinforcement learning enviroment -> Agent
 100 | """
 101 | 
 102 | class MarketExperiment5(TorchExperiment):
 103 | 
 104 |     def __init__(self, *args, **kwargs):
 105 |         super(MarketExperiment5, self).__init__(*args, **kwargs)
 106 |         self.model: Optional[MarketAgent] = None
 107 |         self.teacher_model: Optional[MarketAgent] = None
 108 |         self.teacher_ensemble = {}
 109 |         self.optimizer = None
 110 | 
 111 |     def eval(self, data, from_checkpoint: Optional[int] = None,
 112 |             show_progress=True, batch_size=3000, warmup_window=10,
 113 |             from_date=None, to_date=None, pairs=None, dir=None):
 114 | 
 115 |         if self.model is None or isinstance(from_checkpoint, int):
 116 |             checkpoints = filter(lambda p: 'checkpoint' in p.stem and p.is_dir(), self.exp_path.iterdir())
 117 | 
 118 |             # last_checkpoint = sorted(checkpoints, key=lambda p: int(p.stem.split("_")[1]))[-1 or from_checkpoint]
 119 |             # exp_state_dict = torch.load(str(last_checkpoint / 'exp_state_dict.pkl'), map_location=self.device)
 120 |             exp_state_dict = torch.load(dir , map_location=self.device)
 121 |             model = MarketAgent(**self.db['model_params'])
 122 |             model.load_state_dict(exp_state_dict['model_state_dict'])
 123 |             model.to(self.device)
 124 |             model.eval()
 125 |             self.model = model
 126 |         else:
 127 |             model = self.model
 128 |         pairs = pairs or list(data['asset_index'].keys())
 129 |         asset_index_dict = {k: data['asset_index'][k] for k in pairs}
 130 |         idxs_ranges = create_pair_sample_ranges(asset_index_dict, freq='3M',
 131 |                                                 from_date=from_date, to_date=to_date)
 132 | 
 133 |         vecenv = VecCandleMarketEnv(auto_reset=False, **data, **self.db['env_params'])
 134 |         env = PytorchConverter(vecenv, device=self.device)
 135 |         n_loops = int(np.ceil(len(idxs_ranges) / batch_size))
 136 |         pbar = tqdm(
 137 |             total=n_loops * max(idx['steps'] for idx in idxs_ranges),
 138 |             desc=f'Running Test Batch 1/{n_loops}. Batch Size {batch_size}',
 139 |             disable=not show_progress)
 140 |         info_list = defaultdict(list)
 141 |         last_reset = 0
 142 |         max_ep_len = self.db['env_params']['max_episode_steps']
 143 |         with torch.no_grad():
 144 |             for i in range(0, n_loops):
 145 |                 batch_idxs = idxs_ranges[i * batch_size: (i + 1) * batch_size]
 146 |                 start_idxs = np.array([v['start'] for v in batch_idxs])
 147 |                 stop_idxs = np.array([v['stop'] for v in batch_idxs])
 148 |                 vecenv.reset(stop_idxs=stop_idxs,
 149 |                             start_idxs=start_idxs,
 150 |                             pairs=[v['pair'] for v in batch_idxs])
 151 |                 state = None
 152 |                 rews, obs, is_first = env.observe()
 153 |                 out = model(state=state, **obs)
 154 |                 state = out['state']
 155 |                 pobs = deque(maxlen=warmup_window)
 156 |                 while True:
 157 |                     env.act(dict(market_action=out['market_action']))
 158 |                     rews, obs, is_first = env.observe()
 159 |                     for k, v in vecenv.get_info().items():
 160 |                         info_list[k].append(v)
 161 |                     pobs.append(obs)
 162 |                     last_reset += 1
 163 |                     if np.any(is_first):
 164 |                         if np.all(is_first):
 165 |                             break
 166 |                         vecenv.drop_envs(is_first)
 167 |                         for pi, obs_ in enumerate(pobs):
 168 |                             if any(is_first):
 169 |                                 for k in obs.keys():
 170 |                                     obs_[k] = obs_[k][~is_first]
 171 |                         pbar.desc = f'Running Test Batch {i + 1}/{n_loops}. Batch Size {vecenv.num}'
 172 |                         for state_key, state_value in state.items():
 173 |                             state[state_key] = [st[:, ~is_first] for st in state_value]
 174 |                     if last_reset > max_ep_len:
 175 |                         for k, v in info_list.items():
 176 |                             info_list[k] = [np.concatenate(v)]
 177 |                         last_reset = 0
 178 |                         state = None
 179 |                         for pi, obs_ in enumerate(pobs):
 180 |                             pobs[pi] = obs_
 181 |                             out = model(state=state, **obs_)
 182 |                             state = out['state']
 183 |                     else:
 184 |                         out = model(state=state, **obs)
 185 |                         state = out['state']
 186 |                         pbar.update(1)
 187 | 
 188 |         pbar.close()
 189 |         info_list = {k: np.concatenate(v) for k, v in info_list.items()}
 190 |         df = pd.DataFrame.from_dict(info_list)
 191 |         del info_list
 192 |         res_dict = dict()
 193 |         for pair_encoding, df in df.groupby('pair_encoding'):
 194 |             df = df.drop('pair_encoding', axis=1)
 195 |             df.set_index('time_idx', inplace=True)
 196 |             df = df.iloc[~df.index.duplicated(keep='first')]
 197 |             df.sort_index(inplace=True)
 198 |             pair = vecenv.pairs[pair_encoding]
 199 |             res_dict[pair] = df.iloc[:-1]
 200 |         return res_dict
 201 | 
 202 |     """
 203 |     :param data: training data with columns trade price, transaction price, position (-1, 0, 1), reward
 204 |     :param res_dict: evaluation data (same as data parameter)
 205 |     :train_end: end date training i.e 2021
 206 | 
 207 |     :return: a dictionary with keys for each asset i.e Bitcoin. Eth etc. For each asset we get the training pnl and the
 208 |     testing pnl with the excact date  and the pnl for this date i.e
 209 |     'BTCUSDT': {'train_pnl':
 210 |         2017-08-17 21:00:00    0.0,
 211 |         2017-08-17 22:00:00    0.0 
 212 |             ...
 213 |             ...
 214 |             }
 215 |             ...       
 216 |     """
 217 | 
 218 |     def detailed_backtest(self, data, res_dict, train_end):
 219 |         candle_dict, asset_index = data['candle_dict'], data['asset_index']
 220 |         pnl_ranges = dict()
 221 |         train_end = pd.to_datetime(train_end)
 222 | 
 223 |         for k in res_dict.keys():
 224 |             assert np.allclose(candle_dict[k].shape[0], res_dict[k].shape[0], atol=10, rtol=0)
 225 |             candles, trade_price, positions = candle_dict[k], res_dict[k]['trade_price'], res_dict[k]['position']
 226 |             candles = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close'], index=data['asset_index'][k])
 227 |             pnl = pnl_from_price_position(candles, trade_price=trade_price,
 228 |                                         positions=positions,
 229 |                                         commission=self.db['env_params']['commission_punishment'])
 230 |             train_end_idx = pnl.index.searchsorted(train_end)
 231 |             pnl_ranges[k] = dict(train_pnl=pnl.iloc[:train_end_idx],
 232 |                                 test_pnl=pnl.iloc[train_end_idx:])
 233 |         return pnl_ranges
 234 | 
 235 |     def backtest(self, data, res_dict, train_end, commission=None):
 236 |         candle_dict, asset_index = data['candle_dict'], data['asset_index']
 237 |         sum_ranges = dict()
 238 |         train_end = pd.to_datetime(train_end)
 239 |         for k in res_dict.keys():
 240 |             assert np.allclose(candle_dict[k].shape[0], res_dict[k].shape[0], atol=10, rtol=0)
 241 |             candles, trade_price, positions = candle_dict[k], res_dict[k]['trade_price'], res_dict[k]['position']
 242 |             candles = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close'], index=data['asset_index'][k])
 243 |             if commission is None:
 244 |                 pnl = pnl_from_price_position(candles, trade_price=trade_price,
 245 |                                             positions=positions,
 246 |                                             commission=self.db['env_params']['commission_punishment'])
 247 |             else:
 248 |                 pnl = pnl_from_price_position(candles, trade_price=trade_price,
 249 |                                             positions=positions,
 250 |                                             commission=commission)
 251 |             train_end_idx = pnl.index.searchsorted(train_end)
 252 |             sum_ranges[k] = dict(train_pnl=pnl.iloc[:train_end_idx].sum(),
 253 |                                 test_pnl=pnl.iloc[train_end_idx:].sum())
 254 |         global_sum_range = dict(train_pnl=sum([sr['train_pnl'] for sr in sum_ranges.values()]),
 255 |                                 test_pnl=sum([sr['test_pnl'] for sr in sum_ranges.values()]))
 256 | 
 257 |         print("Back Testing")
 258 |         print(global_sum_range)
 259 |         return global_sum_range
 260 | 
 261 | 
 262 | 
 263 |     """
 264 |     Create different seeds -> for our model
 265 |     """
 266 |     def _fix_random_seed(self, manual_seed):
 267 |         # Fix seed
 268 |         random.seed(manual_seed)
 269 |         torch.manual_seed(manual_seed)
 270 |         torch.cuda.manual_seed(manual_seed)
 271 |         np.random.seed(manual_seed)
 272 |         torch.backends.cudnn.deterministic = True
 273 | 
 274 |         return manual_seed
 275 | 
 276 |     def repackage_hidden(self, h):
 277 |         """Wraps hidden states in new Variables, to detach them from their history."""
 278 |         if type(h) == Variable:
 279 |             return Variable(h.data)
 280 |         else:
 281 |             return tuple(self.repackage_hidden(v) for v in h)
 282 | 
 283 |     def visualize_results(self, data, title):
 284 |         # for i, res in enumerate(data):
 285 |         #     plt.plot(res, '-x', label=legend[i])
 286 |         sns.set_theme()
 287 |         plt.plot(data, '-x')
 288 |         # plt.xticks(np.arange(0, epoch, 1))
 289 |         # plt.plot(data)
 290 |         plt.xlabel('No. of epoch')
 291 |         # plt.ylabel(ylabel)
 292 |         plt.title(title)
 293 |         plt.show()
 294 | 
 295 |     def _compute_distillation_loss(self, student_eval: Dict, teacher_eval: Dict, **kwargs) -> torch.Tensor:
 296 |         """
 297 |         Hook for computing the distillation loss.
 298 |         In the parent class, this computes the PKT loss.
 299 |         Child classes can override this to implement different distillation methods.
 300 |         
 301 |         Args:
 302 |             student_eval: Student model evaluation results including logits
 303 |             teacher_eval: Teacher model evaluation results (unused in PKT)
 304 |             **kwargs: Additional parameters needed for distillation calculation
 305 |             
 306 |         Returns:
 307 |             Weighted distillation loss
 308 |         """
 309 |         # Extract required parameters from kwargs
 310 |         features = kwargs.get('features')
 311 |         position = kwargs.get('position')
 312 |         actions = kwargs.get('actions')
 313 |         batch_size = kwargs.get('batch_size')
 314 |         student_features = kwargs.get('student_features')
 315 |         
 316 |         pkt_losses = []
 317 |         # load the values from ensemble teachers for current actions
 318 |         # and take the mean of the output
 319 |         for name, models in self.teacher_ensemble.items():
 320 |             with torch.no_grad():
 321 |                 teacher_features = models.get_features(actions=actions, features=features,
 322 |                                                     position=position)
 323 | 
 324 |                 # q(a|s) = res['logits'] -> teacher output distribution
 325 |                 # p(a|s) = train_eval['logits'] -> student output distribution
 326 |                 # cross_entropy(
 327 |                 pkt_losses.append(
 328 |                     cosine_similarity_loss(student_features.reshape(batch_size, 40 * 42),
 329 |                                         teacher_features.reshape(batch_size, 40 * 42)))
 330 | 
 331 |         # average of ensemble
 332 |         tensor1 = torch.tensor(pkt_losses, requires_grad=True)
 333 |         pkt_loss = torch.mean(tensor1)
 334 |         
 335 |         return pkt_loss
 336 | 
 337 |     def _student_train_internal(self, data, model_params=None, env_params=None, n_epochs=100, batch_size=32, ppo_clip=0.2,
 338 |                     entropy_weight=0.01, train_range=("2000", "2016"), show_progress=True, gamma=0.9, tau=0.95,
 339 |                     value_horizon=np.inf, lr=5e-4, validation_interval=100, n_envs=128, n_reuse_policy=3,
 340 |                     n_reuse_value=1,
 341 |                     n_reuse_aux=0, weight_decay=0., checkpoint_dir=None, advantage_type='hyperbolic', use_amp=False,
 342 |                     env_step_init=1.0, rew_limit=6., recompute_values=False,
 343 |                     truncate_bptt=(5, 20), lookahead=False, teacher_output=None, seed=0, **kwargs):
 344 |         """
 345 |         Student training for PKT (Probabilistic Knowledge Transfer) method.
 346 |         
 347 |         Args:
 348 |             **kwargs: Additional parameters that may be passed from orchestrators.
 349 |                      These are safely ignored if not used by this specific training method.
 350 |         """
 351 |         #
 352 |         # if 'model_params' in self.db and model_params is not None:
 353 |         #     self.logger.warning("model_params is already set. New Values ignored.")
 354 |         # self.db['model_params'] = model_params = self.db.get('model_params', model_params)
 355 |         #
 356 |         if 'env_params' in self.db and env_params is not None:
 357 |             self.logger.warning("env_params is already set. New Values ignored.")
 358 |         self.db['env_params'] = env_params = self.db.get('env_params', env_params)
 359 | 
 360 |         print(model_params)
 361 |         # Define the model -> Trading Agent
 362 |         self.model = None
 363 |         if self.model is None:
 364 |             num_inputs = list(data['feature_dict'].values())[0].shape[1]
 365 |             model_params['num_inputs'] = num_inputs
 366 |             self.db['model_params'] = model_params
 367 |             manual_seed = self._fix_random_seed(1 + seed)
 368 | 
 369 |             model = MarketAgent(**model_params)
 370 |             model.to(self.device)
 371 |             self.model = model
 372 | 
 373 |         # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
 374 |         optimizer = RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
 375 |         if lookahead:
 376 |             optimizer = Lookahead(optimizer)
 377 |         self.optimizer = optimizer
 378 | 
 379 |         aux_optimizer = RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
 380 |         # if lookahead:
 381 |         #     aux_optimizer = Lookahead(aux_optimizer)
 382 | 
 383 |         # lr_decay_lambda = create_hyperbolic_decay(
 384 |         #     1.0, 0.1, int(n_epochs * 0.8), hyperbolic_steps=10
 385 |         # )
 386 | 
 387 |         lr_decay_lambda = create_hyperbolic_decay(
 388 |             1, 0.1, int(n_epochs * 0.8), hyperbolic_steps=10
 389 |         )
 390 | 
 391 |         lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_decay_lambda)
 392 |         aux_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(aux_optimizer, lr_lambda=lr_decay_lambda)
 393 | 
 394 |         return_stats = RunningMeanStd()
 395 |         cur_train_step = 0
 396 |         if checkpoint_dir:
 397 |             checkpoint_dict = self.load_checkpoint(checkpoint_dir, MarketAgent, load_rng_states=False)
 398 |             lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler_state_dict'])
 399 |             cur_train_step = checkpoint_dict.get('cur_train_step', 0)
 400 |             return_stats = checkpoint_dict.get('return_stats', RunningMeanStd())
 401 | 
 402 |         summary_writer = SummaryWriter(str(self.exp_path), max_queue=500)
 403 |         env = VecCandleMarketEnv(
 404 |             n_envs=n_envs,
 405 |             sample_range=train_range,
 406 |             **env_params,
 407 |             **data)
 408 |         env = PytorchConverter(env, device=self.device)
 409 |         env = NormalizeWrapper(env, clip_reward=rew_limit)
 410 | 
 411 |         gae_engine = NormalizedGAE(advantage_type=advantage_type,
 412 |                                 return_stats=return_stats,
 413 |                                 device=self.device)
 414 |         cross_entropy = MineCrossEntropy(dim=1)
 415 |         # soft_max = nn.Softmax(dim=-1)
 416 |         training_loss = []
 417 | 
 418 |         # print(self.teacher_ensemble)
 419 |         #
 420 |         # print(self.model)
 421 |         policy_loss_history = []
 422 |         value_loss_history = []
 423 |         distillation_loss_history = []
 424 |         entropy_history = []
 425 |         cumulative_reward_history = []
 426 |         # Training loop
 427 |         for gi in tqdm(range(cur_train_step, n_epochs), smoothing=0., disable=not show_progress):
 428 |             trajectories = gather_trajectories(env, model, include_value=not recompute_values)
 429 | 
 430 |             rew_array, obs_tensor_dict, old_log_prob_tensor, action_tensor, old_logits, *value_tensor = \
 431 |                 prepare_trajectory_windows(trajectories)
 432 |             value_tensor = None if not value_tensor else value_tensor[0].squeeze()
 433 | 
 434 |             metrics_dict = dict(
 435 |                 rew_std=np.sqrt(env.rew_stats.var),
 436 |                 mean_reward=rew_array.sum(axis=1).mean(),
 437 |                 mean_action=action_tensor.float().mean().detach().cpu(),
 438 |             )
 439 |             if value_tensor is not None:
 440 |                 metrics_dict['mean_value'] = value_tensor.mean()
 441 | 
 442 |             if (gi % 50) == 0:
 443 |                 summary_writer.add_histogram('train/action_dist', action_tensor.detach(), gi, bins=5)
 444 |                 summary_writer.add_histogram('train/value_dist', value_tensor.flatten(), gi, max_bins=60)
 445 | 
 446 |             model.train()
 447 | 
 448 |             # Policy Train
 449 |             avg_train_metrics = defaultdict(RunningScalarMean)
 450 |             hist_train_metrics = defaultdict(list)
 451 |             advantage_array, returns_array = gae_engine.advantage(
 452 |                 rew_array, value_tensor, gamma=gamma, tau=tau, horizon=value_horizon, clip=1.)
 453 |             for ti in range(n_reuse_policy):
 454 |                 # load these parameters in teacher model too, for distillation
 455 |                 for bi, (actions, old_log_prob, features, position, rews, old_logits_b, value_pred, advantage,
 456 |                         returns) in enumerate(
 457 |                     random_batched_iterator(
 458 |                         action_tensor, old_log_prob_tensor, obs_tensor_dict['features'],
 459 |                         obs_tensor_dict['position'], rew_array, old_logits, value_tensor,
 460 |                         advantage_array, returns_array,
 461 |                         batch_size=batch_size,
 462 |                     )):
 463 |                     optimizer.zero_grad(set_to_none=True)
 464 |                     train_eval = model.train_eval(actions=actions, features=features,
 465 |                                                 position=position)
 466 |                     student_features = model.get_features(actions=actions, features=features,
 467 |                                                         position=position)
 468 |                     # value_pred = to_np(train_eval['value'].squeeze()).astype(np.float32)
 469 |                     # advantage, returns = gae_engine.advantage(rews, value_pred)
 470 |                     policy_loss, pstats = ppo_categorical_policy_loss(actions, old_log_prob,
 471 |                                                                     Categorical(logits=old_logits_b),
 472 |                                                                     train_eval["pd"], advantage, ppo_clip=ppo_clip)
 473 | 
 474 |                     if entropy_weight > 0:
 475 |                         raw_entropy = train_eval["pd"].entropy().mean()
 476 |                         entropy_loss = -entropy_weight * train_eval["pd"].entropy().mean()
 477 |                         policy_loss += entropy_loss
 478 |                         avg_train_metrics['entropy_loss'].update(to_np(entropy_loss))
 479 |                         avg_train_metrics['policy_entropy'].update(to_np(raw_entropy))
 480 | 
 481 |                     value_loss = torch.pow(train_eval['value'].squeeze(2) - returns, 2)[:, :-10].mean()
 482 | 
 483 |                     # Use the hook method for distillation loss computation
 484 |                     # This allows child classes to override with different distillation methods
 485 |                     distillation_loss = self._compute_distillation_loss(
 486 |                         train_eval, {}, # Empty teacher_eval for PKT since it uses ensemble
 487 |                         features=features, position=position, actions=actions, 
 488 |                         batch_size=batch_size, student_features=student_features
 489 |                     )
 490 | 
 491 |                     loss = policy_loss + value_loss + distillation_loss
 492 | 
 493 |                     avg_train_metrics['value_loss'].update(to_np(value_loss))
 494 |                     avg_train_metrics['policy_loss'].update(to_np(policy_loss))
 495 |                     avg_train_metrics['distillation_loss'].update(to_np(distillation_loss))
 496 |                     hist_train_metrics['advantage'].append(advantage.detach().flatten())
 497 |                     loss.backward()
 498 |                     nn.utils.clip_grad_norm_(model.parameters(), 1.)
 499 |                     optimizer.step()
 500 |             if n_reuse_aux > 0:
 501 |                 with torch.no_grad():
 502 |                     new_train_eval = model.train_eval(obs_tensor_dict['features'], obs_tensor_dict['position'],
 503 |                                                     action_tensor)
 504 |                     new_train_eval['logits'].detach_()
 505 |                 if lookahead:
 506 |                     optimizer.slow_step()
 507 |             # Aux Train
 508 |             for ti in range(n_reuse_aux):
 509 |                 for bi, (actions, features, position, rews, new_logits, value_pred, advantage, returns) in enumerate(
 510 |                         random_batched_iterator(
 511 |                             action_tensor, obs_tensor_dict['features'], obs_tensor_dict['position'],
 512 |                             rew_array, new_train_eval['logits'], value_tensor,
 513 |                             advantage_array, returns_array,
 514 |                             batch_size=batch_size,
 515 |                         )):
 516 |                     aux_optimizer.zero_grad(set_to_none=True)
 517 |                     train_eval = model.train_eval(actions=actions, features=features,
 518 |                                                 position=position, truncate_bptt=truncate_bptt,
 519 |                                                 policy_aux_val=True)
 520 |                     # advantage, returns = gae_engine.advantage(rews, value_pred)
 521 |                     value_loss = torch.pow(train_eval['aux_val_pred'].squeeze(2) - returns, 2).mean()
 522 |                     clone_loss = kl_divergence(Categorical(logits=new_logits), train_eval['pd']).mean()
 523 |                     loss: torch.Tensor = value_loss + 20. * clone_loss
 524 |                     avg_train_metrics['aux_val_loss'].update(to_np(value_loss))
 525 |                     avg_train_metrics['aux_clone_loss'].update(to_np(clone_loss))
 526 |                     loss.backward()
 527 |                     nn.utils.clip_grad_norm_(model.parameters(), 1.)
 528 |                     aux_optimizer.step()
 529 |             aux_lr_scheduler.step()
 530 |             lr_scheduler.step()
 531 |             avg_train_metrics = {k: v.mean for k, v in avg_train_metrics.items()}
 532 |             
 533 |             policy_loss_history.append(avg_train_metrics.get('policy_loss', 0))
 534 |             value_loss_history.append(avg_train_metrics.get('value_loss', 0))
 535 |             distillation_loss_history.append(avg_train_metrics.get('distillation_loss', 0))
 536 |             entropy_history.append(avg_train_metrics.get('policy_entropy', 0))
 537 |             cumulative_reward_history.append(metrics_dict.get('mean_reward', 0))
 538 |             
 539 |             if ((gi + 1) % validation_interval) == 0 or gi==n_epochs-1:
 540 |                 self.checkpoint(global_step=gi, return_stats=return_stats,
 541 |                                 lr_scheduler_state_dict=lr_scheduler.state_dict(),
 542 |                                 cur_train_step=gi)
 543 |                 # eval_dict = self.eval(data, show_progress=show_progress)
 544 |                 # cur_performance = self.backtest(data, eval_dict, train_end=train_range[1])
 545 |                 # # cur_performance_pnl = self.backtest(data, eval_dict, train_end=train_range[1], commission=1e-3)
 546 |                 # cur_performance_pnl = self.backtest(data, eval_dict, train_end=train_range[1], commission=2e-5)
 547 | 
 548 | 
 549 |                 # num_trades_test = 1
 550 |                 # num_trades_train = 1
 551 |                 # num_exit_test = 0
 552 |                 # num_pos_test = 0
 553 |                 # num_exit_train = 0
 554 |                 # num_pos_train = 0
 555 |                 # for k in eval_dict.keys():
 556 |                 #     train = eval_dict[k][eval_dict[k].index < pd.to_datetime(train_range[1])].position.diff()
 557 |                 #     test = eval_dict[k][eval_dict[k].index >= pd.to_datetime(train_range[1])].position.diff()
 558 |                 #     num_trades_train += train[train != 0].shape[0]
 559 |                 #     num_trades_test += test[test != 0].shape[0]
 560 |                 #     exit_train = eval_dict[k][eval_dict[k].index < pd.to_datetime(train_range[1])].position
 561 |                 #     num_pos_train += exit_train.shape[0]
 562 |                 #     num_exit_train += exit_train[exit_train == 0].shape[0]
 563 |                 #     exit_test = eval_dict[k][eval_dict[k].index >= pd.to_datetime(train_range[1])].position
 564 |                 #     num_exit_test += exit_test[exit_test == 0].shape[0]
 565 |                 #     num_pos_test += exit_test.shape[0]
 566 |                 # num_trades_test /= len(eval_dict.keys())
 567 |                 # num_trades_train /= len(eval_dict.keys())
 568 |                 # num_exit_train /= num_pos_train
 569 |                 # num_exit_test /= num_pos_test
 570 |                 if tune.is_session_enabled() is not None:
 571 |                     # for k, v in cur_performance.items():
 572 |                     #     summary_writer.add_scalar(f'eval/{k}', v, gi)
 573 |                     # for k, v in cur_performance_pnl.items():
 574 |                     #     summary_writer.add_scalar(f'pnl_/{k}', v, gi)
 575 |                     for k, v in metrics_dict.items():
 576 |                         summary_writer.add_scalar(f'metrics/{k}', v, gi)
 577 |                     for k, v in model.weight_stats().items():
 578 |                         summary_writer.add_scalar(f'model/{k}', v, gi)
 579 |                     for k, v in avg_train_metrics.items():
 580 |                         summary_writer.add_scalar(f'train/{k}', v, gi)
 581 |                     summary_writer.add_scalar(f'train/lr', lr_scheduler.get_last_lr()[0], gi)
 582 |                     # summary_writer.add_scalar(f'trades/train_trades', num_trades_train, gi)
 583 |                     # summary_writer.add_scalar(f'trades/test_trades', num_trades_test, gi)
 584 |                     # summary_writer.add_scalar(f'trades/train_exit', num_exit_train, gi)
 585 |                     # summary_writer.add_scalar(f'trades/test_exit', num_exit_test, gi)
 586 |                 else:
 587 |                     tune.report(global_step=gi, lr=lr_scheduler.get_last_lr()[0],
 588 |                                  **metrics_dict, **model.weight_stats(),
 589 |                                 **avg_train_metrics
 590 |                                 )
 591 |                 # if ((gi + 1) % 5000) == 0:
 592 |                 #     detailed_pnls = self.detailed_backtest(data, eval_dict, train_end=train_range[1])
 593 |                 #     fig = go.FigureWidget()
 594 |                 #     for pair, pnls in detailed_pnls.items():
 595 |                 #         train_pnl = pnls['train_pnl'].cumsum()
 596 |                 #         test_pnl = train_pnl.values[-1] + pnls['test_pnl'].cumsum()
 597 |                 #         fig.add_scatter(x=train_pnl.index, y=train_pnl, legendgroup=pair, name=pair)
 598 |                 #         fig.add_scatter(x=test_pnl.index, y=test_pnl, legendgroup=pair, name=pair)
 599 |                 #     fig.write_html(f'plots/figure_{lr}_{gi+1}.html')
 600 |                 #     del detailed_pnls
 601 |                 # del eval_dict
 602 |         
 603 |         # Compile comprehensive training metrics for academic reporting
 604 |         final_training_report = {
 605 |             'final_train_metrics': avg_train_metrics, # The metrics from the final epoch
 606 |             'model_stats': model.weight_stats(),
 607 |             'final_lr': lr_scheduler.get_last_lr()[0],
 608 |             'total_epochs': n_epochs,
 609 |             # History lists for academic analysis
 610 |             'policy_loss_history': policy_loss_history,
 611 |             'value_loss_history': value_loss_history,
 612 |             'distillation_loss_history': distillation_loss_history,
 613 |             'policy_entropy_history': entropy_history,
 614 |             'cumulative_reward_history': cumulative_reward_history,
 615 |         }
 616 | 
 617 |         return final_training_report
 618 |     
 619 |     def student_train(self, data, model_params=None, env_params=None, **training_params):
 620 |         """
 621 |         PUBLIC-FACING training method. Calls the internal training loop and
 622 |         returns a standardized dictionary of training-related metrics.
 623 |         """
 624 |         # Delegate to the internal training loop
 625 |         training_report = self._student_train_internal(data, model_params, env_params, **training_params)
 626 | 
 627 |         # Create the clean, standardized dictionary for the orchestrator
 628 |         structured_metrics = {
 629 |             'final_train_metrics': training_report.get('final_train_metrics', {}),
 630 |             'model_stats': training_report.get('model_stats', {}),
 631 |             'total_epochs': training_report.get('total_epochs', 0),
 632 |             'method': 'Offline_PKT', # Hardcode the method name for this class
 633 |             'distillation_type': 'ensemble_probabilistic_knowledge_transfer',
 634 |             # Include training history data for academic reporting pipeline
 635 |             'policy_loss_history': training_report.get('policy_loss_history', []),
 636 |             'value_loss_history': training_report.get('value_loss_history', []),
 637 |             'distillation_loss_history': training_report.get('distillation_loss_history', []),
 638 |             'policy_entropy_history': training_report.get('policy_entropy_history', []),
 639 |             'cumulative_reward_history': training_report.get('cumulative_reward_history', [])
 640 |         }
 641 |         return structured_metrics
 642 |     def train_and_save_ensemble(self, data: dict, output_dir: Path, teacher_model_params: dict, env_params: dict, 
 643 |                                 n_teachers: int, seed: int, **training_kwargs):
 644 |         """
 645 |         Trains an ensemble of teacher models and saves each one to a file. This is a standalone
 646 |         process for generating reusable teacher assets.
 647 | 
 648 |         Args:
 649 |             data (dict): Market data dictionary.
 650 |             output_dir (Path): The directory where trained teacher models will be saved.
 651 |             teacher_model_params (dict): Parameters for the teacher models.
 652 |             env_params (dict): Parameters for the training environment.
 653 |             n_teachers (int): The number of teachers to train.
 654 |             seed (int): The base random seed for ensuring diversity.
 655 |             **training_kwargs: All other training parameters (lr, n_epochs, etc.).
 656 |         """
 657 |         print(f"--- Starting Standalone Teacher Ensemble Training ---")
 658 |         print(f"Number of teachers: {n_teachers} | Output directory: {output_dir}")
 659 |         output_dir.mkdir(parents=True, exist_ok=True)
 660 | 
 661 |        
 662 |         num_inputs = list(data['feature_dict'].values())[0].shape[1]
 663 |         teacher_model_params['num_inputs'] = num_inputs
 664 |         
 665 |         teachers_to_train = {}
 666 |         for i in range(n_teachers):
 667 |             self._fix_random_seed(seed + i)
 668 |             model = MarketAgent(**teacher_model_params)
 669 |             model.to(self.device)
 670 |             teachers_to_train[f'teacher_{i+1}'] = model
 671 |         
 672 |         print(f"Initialized {len(teachers_to_train)} unique teacher models.")
 673 | 
 674 |         for name, model in teachers_to_train.items():
 675 |             print(f"\n--- Training {name} ---")
 676 |             self._train_single_teacher(model, data, env_params, **training_kwargs)
 677 | 
 678 |             save_path = output_dir / f"{name}.pt"
 679 |             torch.save({
 680 |                 'model_state_dict': model.state_dict(),
 681 |                 'teacher_name': name,
 682 |                 'model_params': teacher_model_params,
 683 |                 'training_info': {
 684 |                     'n_epochs': training_kwargs.get('n_epochs', 100),
 685 |                     'seed': seed + int(name.split('_')[1]) - 1,
 686 |                     'final_loss': total_loss.item() if 'total_loss' in locals() else None
 687 |                 }
 688 |             }, save_path)
 689 |             print(f"--- Saved trained {name} to {save_path} ---")
 690 | 
 691 |         print("\n--- Standalone Teacher Ensemble Training Finished ---")
 692 |         return [output_dir / f"teacher_{i+1}.pt" for i in range(n_teachers)]
 693 | 
 694 |     def _train_single_teacher(self, model, data, env_params, **training_kwargs):
 695 |         """
 696 |         Helper method to train a single teacher model with full PPO training loop.
 697 |         Extracts the training logic to properly use all training parameters.
 698 |         
 699 |         Args:
 700 |             model: The teacher model to train
 701 |             data: Training data
 702 |             env_params: Environment parameters
 703 |             **training_kwargs: All training parameters (lr, gamma, tau, etc.)
 704 |         """
 705 |         optimizer = RAdam(model.parameters(), lr=training_kwargs.get('lr', 5e-4))
 706 |         
 707 |         env = VecCandleMarketEnv(n_envs=training_kwargs.get('n_envs', 128), **env_params, **data)
 708 |         env = PytorchConverter(env, device=self.device)
 709 |         env = NormalizeWrapper(env, clip_reward=training_kwargs.get('rew_limit', 6.0))
 710 |         gae_engine = NormalizedGAE(advantage_type=training_kwargs.get('advantage_type', 'hyperbolic'),
 711 |                                   return_stats=RunningMeanStd(),
 712 |                                   device=self.device)
 713 | 
 714 |         for gi in tqdm(range(training_kwargs.get('n_epochs', 100)), desc=f"Training teacher", smoothing=0.):
 715 |             trajectories = gather_trajectories(env, model, include_value=True)
 716 |             rew_array, obs_dict, old_log_probs, actions, old_logits, values = prepare_trajectory_windows(trajectories)
 717 |             
 718 |             model.train()
 719 |             advantages, returns = gae_engine.advantage(rew_array, values.squeeze(), 
 720 |                                                      gamma=training_kwargs.get('gamma', 0.99), 
 721 |                                                      tau=training_kwargs.get('tau', 0.95))
 722 |             
 723 |             for _ in range(training_kwargs.get('n_reuse_policy', 3)):
 724 |                 for batch_data in random_batched_iterator(actions, old_log_probs, obs_dict['features'], 
 725 |                                                          obs_dict['position'], advantages, returns, old_logits, 
 726 |                                                          batch_size=training_kwargs.get('batch_size', 32)):
 727 |                     b_actions, b_old_log_probs, b_features, b_pos, b_adv, b_returns, b_old_logits = batch_data
 728 |                     
 729 |                     optimizer.zero_grad()
 730 |                     train_eval = model.train_eval(actions=b_actions, features=b_features, position=b_pos)
 731 |                     policy_loss, _ = ppo_categorical_policy_loss(b_actions, b_old_log_probs, Categorical(logits=b_old_logits),
 732 |                                                                  train_eval["pd"], b_adv, ppo_clip=training_kwargs.get('ppo_clip', 0.2))
 733 |                     value_loss = torch.pow(train_eval['value'].squeeze() - b_returns, 2).mean()
 734 |                     
 735 |                     # Add entropy loss if specified
 736 |                     total_loss = policy_loss + value_loss
 737 |                     if training_kwargs.get('entropy_weight', 0) > 0:
 738 |                         entropy_loss = -training_kwargs.get('entropy_weight', 0) * train_eval["pd"].entropy().mean()
 739 |                         total_loss += entropy_loss
 740 |                     
 741 |                     total_loss.backward()
 742 |                     nn.utils.clip_grad_norm_(model.parameters(), 1.0)
 743 |                     optimizer.step()
 744 | 
 745 |     def load_teacher_ensemble(self, teacher_paths: list, teacher_model_params: dict) -> dict:
 746 |         """
 747 |         Loads a pre-trained teacher ensemble from saved model files.
 748 | 
 749 |         Args:
 750 |             teacher_paths: List of paths to saved teacher model files.
 751 |             teacher_model_params: Parameters for creating teacher model instances.
 752 | 
 753 |         Returns:
 754 |             Dictionary mapping teacher names to loaded models.
 755 |         """
 756 |         print(f"--- Loading Teacher Ensemble from {len(teacher_paths)} models ---")
 757 |         
 758 |         num_inputs = teacher_model_params.get('num_inputs')
 759 |         if num_inputs is None:
 760 |             raise ValueError("teacher_model_params must include 'num_inputs'")
 761 |         
 762 |         teacher_ensemble = {}
 763 |         
 764 |         for i, teacher_path in enumerate(teacher_paths):
 765 |             if not Path(teacher_path).exists():
 766 |                 print(f"Warning: Teacher model not found at {teacher_path}, skipping...")
 767 |                 continue
 768 |             
 769 | 
 770 |             checkpoint = torch.load(teacher_path, map_location=self.device)
 771 |             
 772 |             teacher_name = checkpoint.get('teacher_name', f'teacher_{i+1}')
 773 |             model = MarketAgent(**teacher_model_params)
 774 |             model.load_state_dict(checkpoint['model_state_dict'])
 775 |             model.to(self.device)
 776 |             model.eval() 
 777 |             
 778 |             teacher_ensemble[teacher_name] = model
 779 |             print(f"Loaded teacher: {teacher_name} from {teacher_path}")
 780 |         
 781 |         self.teacher_ensemble = teacher_ensemble
 782 |         print(f"Successfully loaded {len(teacher_ensemble)} teachers into ensemble")
 783 |         return teacher_ensemble
 784 | 
 785 |     def select_best_teachers(self, data: dict, teacher_ensemble: dict, eval_config: dict,
 786 |                            n_select: int = 3) -> list:
 787 |         """
 788 |         Evaluates all teachers and selects the top N performing ones.
 789 | 
 790 |         Args:
 791 |             data: Market data for evaluation.
 792 |             teacher_ensemble: Dictionary of teacher models.
 793 |             eval_config: Configuration for evaluation.
 794 |             n_select: Number of best teachers to select.
 795 | 
 796 |         Returns:
 797 |             List of teacher names representing the best performers.
 798 |         """
 799 |         print(f"--- Evaluating {len(teacher_ensemble)} teachers to select best {n_select} ---")
 800 |         
 801 |         teacher_performances = {}
 802 |         
 803 |         # --- FIX: Separate parameters for eval() and backtest() ---
 804 |         
 805 |         # 1. Extract 'train_end' which is needed for backtest, not eval.
 806 |         # We use a default just in case, though the orchestrator should provide it.
 807 |         train_end_date = eval_config.get('train_end', '2021-03-14')
 808 |         
 809 |         # 2. Create a clean dictionary of parameters specifically for the eval() method.
 810 |         eval_params = eval_config.copy()
 811 |         eval_params.pop('train_end', None) # Remove train_end from the dictionary
 812 |         
 813 |         for teacher_name, teacher_model in teacher_ensemble.items():
 814 |             print(f"Evaluating {teacher_name}...")
 815 |             
 816 |             original_model = self.model
 817 |             self.model = teacher_model
 818 |             
 819 |             try:
 820 |                 eval_dict = self.eval(data, **eval_params)
 821 |                 
 822 |                 # --- CORRECTED CALL TO backtest() ---
 823 |                 # Pass the extracted train_end date explicitly.
 824 |                 performance = self.backtest(data, eval_dict, train_end=train_end_date)
 825 |                 
 826 |                 teacher_performances[teacher_name] = performance['test_pnl']
 827 |                 print(f"{teacher_name} Test PnL: {performance['test_pnl']:.4f}")
 828 |                 
 829 |             except Exception as e:
 830 |                 print(f"Error evaluating {teacher_name}: {e}")
 831 |                 teacher_performances[teacher_name] = float('-inf')
 832 |             finally:
 833 |                 # Restore original model
 834 |                 self.model = original_model
 835 |         
 836 |         # Select top N teachers
 837 |         sorted_teachers = sorted(teacher_performances.items(), key=lambda x: x[1], reverse=True)
 838 |         selected_teachers = [name for name, _ in sorted_teachers[:n_select]]
 839 |         
 840 |         print(f"Selected best {len(selected_teachers)} teachers: {selected_teachers}")
 841 |         return selected_teachers
 842 | 
 843 | 
 844 |         return training_loss
 845 | 
 846 | 
 847 | """
 848 | Defining loss of the model 
 849 | L_D  = - 1/N * Σ Σ (q(a_i|s) * log (p (a_i|s)) )
 850 | 
 851 | L_D = cross_entropy(logits(student), logits(teacher))
 852 | 
 853 | Final_loss = policy_loss + values_loss + cross_entropy(train_eval['logits'], teacher_train_eval['logits'])
 854 | = L_RL + L_D
 855 | """
 856 | 
 857 | class MineCrossEntropy(nn.Module):
 858 |     def __init__(self, dim=-1):
 859 |         super(MineCrossEntropy, self).__init__()
 860 |         self.dim = dim
 861 | 
 862 |     def forward(self, q, p):
 863 |         p = p.log_softmax(dim=self.dim)
 864 | 
 865 |         return torch.mean(torch.sum(-q * p, dim=self.dim))
 866 | 
 867 | def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
 868 |     # print(output_net.shape)
 869 |     # print(target_net.shape)
 870 |     # print(output_net)
 871 |     # Normalize each vector by its norm
 872 |     output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
 873 |     output_net = output_net / (output_net_norm + eps)
 874 |     output_net[output_net != output_net] = 0
 875 | 
 876 |     target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
 877 |     target_net = target_net / (target_net_norm + eps)
 878 |     target_net[target_net != target_net] = 0
 879 | 
 880 |     # Calculate the cosine similarity
 881 |     model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
 882 |     target_similarity = torch.mm(target_net, target_net.transpose(0, 1))
 883 | 
 884 |     # Scale cosine similarity to 0..1
 885 |     model_similarity = (model_similarity + 1.0) / 2.0
 886 |     target_similarity = (target_similarity + 1.0) / 2.0
 887 | 
 888 |     # Transform them into probabilities
 889 |     model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
 890 |     target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)
 891 | 
 892 |     # Calculate the KL-divergence
 893 |     loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))
 894 | 
 895 |     return loss

```Project Path: academic_integration.py

Source Tree:

```
academic_integration.py

```

`\\?\C:\Users\George\Source\Github\Thesis\financial-trading-in-RL\experiment_code\academic_integration.py`:

```py
   1 | """
   2 | Academic Integration Module for TTM vs PKT Comparison
   3 | ====================================================
   4 | 
   5 | This module integrates the AcademicEvaluator with the existing TTMPKTComparisonOrchestrator
   6 | to provide comprehensive academic-quality reporting and evaluation.
   7 | 
   8 | The integration follows the Master Engineer's evaluation strategy:
   9 | 1. Financial Performance Analysis (ground truth)
  10 | 2. RL Dynamics Analysis (training process insights)
  11 | 3. Behavioral Statistics (trading strategy characterization)
  12 | """
  13 | 
  14 | import numpy as np
  15 | import pandas as pd
  16 | from typing import Dict, List, Any, Optional
  17 | from pathlib import Path
  18 | from datetime import datetime
  19 | import logging
  20 | 
  21 | from experiment_code.academic_evaluator import AcademicEvaluator
  22 | from experiment_code.rl_analysis_utils import (
  23 |     PolicyEntropy,
  24 |     analyze_entropy_trend,
  25 |     # Add any other dataclasses or functions you will use
  26 | )
  27 | 
  28 | class AcademicIntegration:
  29 |     """
  30 |     Integrates academic evaluation capabilities into the existing training orchestrator.
  31 |     
  32 |     This class serves as a bridge between the TTMPKTComparisonOrchestrator and the
  33 |     AcademicEvaluator, ensuring seamless integration while maintaining backward compatibility.
  34 |     """
  35 |     
  36 |     def __init__(self, orchestrator):
  37 |         """
  38 |         Initialize the academic integration with a reference to the orchestrator.
  39 |         
  40 |         Args:
  41 |             orchestrator: Instance of TTMPKTComparisonOrchestrator
  42 |         """
  43 |         self.orchestrator = orchestrator
  44 |         self.config = orchestrator.config
  45 |         self.base_path = orchestrator.base_path
  46 |         self.config_id = orchestrator.config_id
  47 |         
  48 |         # Create academic output directory
  49 |         self.academic_output_dir = self.base_path / self.config_id / "academic_evaluation"
  50 |         self.academic_output_dir.mkdir(parents=True, exist_ok=True)
  51 |         
  52 |         # Initialize academic evaluator
  53 |         self.evaluator = AcademicEvaluator(self.academic_output_dir)
  54 | 
  55 |         logging.basicConfig(level=logging.INFO)
  56 |         self.logger = logging.getLogger(__name__)
  57 |     
  58 |     def enhance_experiment_results(self, method_name: str, results: Dict[str, Any]) -> Dict[str, Any]:
  59 |         """
  60 |         Enhance experiment results with additional metrics required for academic evaluation.
  61 |         
  62 |         Args:
  63 |             method_name: Name of the method ('rl_ttm', 'offline_pkt', 'no_distillation')
  64 |             results: Raw experiment results from orchestrator
  65 |             
  66 |         Returns:
  67 |             Enhanced results with additional academic metrics
  68 |         """
  69 |         self.logger.info(f"Enhancing results for {method_name} with academic metrics")
  70 |         
  71 |         enhanced_results = results.copy()
  72 |         
  73 |         # Extract method-level training metrics (for academic approach)
  74 |         method_training_metrics = self._extract_method_training_metrics(results)
  75 |         
  76 |         # Extract method-level financial data (for academic approach)
  77 |         method_financial_data = self._extract_method_financial_data(results)
  78 |         
  79 |         # Enhance each run with additional metrics
  80 |         enhanced_runs = []
  81 |         for run_idx, run in enumerate(results.get('runs', [])):
  82 |             enhanced_run = self._enhance_single_run(method_name, run, run_idx, method_training_metrics, method_financial_data)
  83 |             enhanced_runs.append(enhanced_run)
  84 |         
  85 |         enhanced_results['runs'] = enhanced_runs
  86 |         
  87 |         # Add method-level academic metrics
  88 |         enhanced_results['academic_metrics'] = self._calculate_method_academic_metrics(
  89 |             method_name, enhanced_runs
  90 |         )
  91 |         
  92 |         return enhanced_results
  93 |     
  94 |     def _enhance_single_run(self, method_name: str, run: Dict[str, Any], run_idx: int, method_training_metrics: Dict[str, Any] = None, method_financial_data: Dict[str, Any] = None) -> Dict[str, Any]:
  95 |         """
  96 |         Enhance a single run with academic metrics.
  97 |         
  98 |         Args:
  99 |             method_name: Name of the method
 100 |             run: Single run results
 101 |             run_idx: Index of the run
 102 |             method_training_metrics: Training metrics at method level (for academic approach)
 103 |             
 104 |         Returns:
 105 |             Enhanced run with additional metrics
 106 |         """
 107 |         enhanced_run = run.copy()
 108 |         
 109 |         # Extract PnL series for financial metrics
 110 |         # Try multiple possible locations for PnL data
 111 |         pnl_series = None
 112 |         
 113 |         # Check for PnL data in various locations
 114 |         possible_pnl_keys = ['avg_test_pnl', 'test_pnl', 'pnl', 'cumulative_pnl', 'returns']
 115 |         for key in possible_pnl_keys:
 116 |             if key in run and run[key] is not None:
 117 |                 pnl_series = run[key]
 118 |                 break
 119 |         
 120 |         # If no PnL in run, check if it's available at method level (academic approach)
 121 |         if pnl_series is None and method_financial_data:
 122 |             for key in possible_pnl_keys:
 123 |                 if key in method_financial_data and method_financial_data[key] is not None:
 124 |                     pnl_series = method_financial_data[key]
 125 |                     break
 126 |         
 127 |         if pnl_series is not None:
 128 |             # Calculate comprehensive financial metrics
 129 |             enhanced_run['financial_metrics'] = self._calculate_financial_metrics(pnl_series)
 130 |             
 131 |             # Calculate risk metrics
 132 |             enhanced_run['risk_metrics'] = self._calculate_risk_metrics(pnl_series)
 133 |         else:
 134 |             # Create empty financial metrics to prevent KeyError
 135 |             enhanced_run['financial_metrics'] = self._create_empty_financial_metrics()
 136 |             enhanced_run['risk_metrics'] = {}
 137 |         
 138 |         # Enhance training metrics for RL dynamics analysis
 139 |         # First check run-level training metrics (legacy approach)
 140 |         training_metrics = run.get('training_metrics', {})
 141 |         
 142 |         # If no run-level training metrics, use method-level training metrics (academic approach)
 143 |         if not training_metrics and method_training_metrics:
 144 |             training_metrics = method_training_metrics
 145 |             
 146 |         if training_metrics:
 147 |             enhanced_run['rl_dynamics'] = self._enhance_rl_dynamics(
 148 |                 method_name, training_metrics, run_idx
 149 |             )
 150 |         
 151 |         # Calculate behavioral metrics from positions
 152 |         eval_dict = run.get('eval_dict', {})
 153 |         if eval_dict:
 154 |             enhanced_run['behavioral_metrics'] = self._calculate_behavioral_metrics(eval_dict)
 155 |         else:
 156 |             # Create empty behavioral metrics to prevent aggregation issues
 157 |             enhanced_run['behavioral_metrics'] = self._create_empty_behavioral_metrics()
 158 |         
 159 |         return enhanced_run
 160 |     
 161 |     def _calculate_financial_metrics(self, pnl_series: pd.Series) -> Dict[str, float]:
 162 |         """Calculate comprehensive financial performance metrics."""
 163 |         if pnl_series is None or len(pnl_series) == 0:
 164 |             return {}
 165 |         
 166 |         # Convert to returns
 167 |         returns = pnl_series.pct_change().dropna()
 168 |         
 169 |         # Basic metrics
 170 |         cumulative_return = pnl_series.iloc[-1] / pnl_series.iloc[0] - 1 if pnl_series.iloc[0] != 0 else 0
 171 |         total_return = pnl_series.iloc[-1]
 172 |         
 173 |         # Risk-adjusted metrics
 174 |         sharpe_ratio = self._calculate_sharpe_ratio(returns)
 175 |         sortino_ratio = self._calculate_sortino_ratio(returns)
 176 |         calmar_ratio = self._calculate_calmar_ratio(pnl_series)
 177 |         
 178 |         # Volatility metrics
 179 |         volatility = returns.std() * np.sqrt(252)  # Annualized
 180 |         
 181 |         return {
 182 |             'cumulative_return': cumulative_return,
 183 |             'total_return': total_return,
 184 |             'sharpe_ratio': sharpe_ratio,
 185 |             'sortino_ratio': sortino_ratio,
 186 |             'calmar_ratio': calmar_ratio,
 187 |             'volatility': volatility,
 188 |             'mean_return': returns.mean(),
 189 |             'std_return': returns.std()
 190 |         }
 191 |     
 192 |     def _calculate_risk_metrics(self, pnl_series: pd.Series) -> Dict[str, float]:
 193 |         """Calculate risk metrics including drawdowns and VaR."""
 194 |         if pnl_series is None or len(pnl_series) == 0:
 195 |             return {}
 196 |         
 197 |         # Calculate drawdowns
 198 |         cumulative = (1 + pnl_series.pct_change().fillna(0)).cumprod()
 199 |         running_max = cumulative.expanding().max()
 200 |         drawdown = (cumulative - running_max) / running_max
 201 |         
 202 |         max_drawdown = abs(drawdown.min())
 203 |         avg_drawdown = abs(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0
 204 |         
 205 |         # Calculate VaR and Expected Shortfall
 206 |         returns = pnl_series.pct_change().dropna()
 207 |         var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
 208 |         var_99 = np.percentile(returns, 1) if len(returns) > 0 else 0
 209 |         
 210 |         # Expected Shortfall (CVaR)
 211 |         es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
 212 |         
 213 |         return {
 214 |             'max_drawdown': max_drawdown,
 215 |             'avg_drawdown': avg_drawdown,
 216 |             'var_95': var_95,
 217 |             'var_99': var_99,
 218 |             'expected_shortfall_95': es_95,
 219 |             'drawdown_duration': self._calculate_drawdown_duration(drawdown)
 220 |         }
 221 |     
 222 |     def _enhance_rl_dynamics(self, method_name: str, training_metrics: Dict, run_idx: int) -> Dict[str, Any]:
 223 |         """
 224 |         Enhance RL dynamics with academic-focused metrics.
 225 |         This version uses the CORRECT data keys from the training output.
 226 |         """
 227 |         enhanced_dynamics = {}
 228 |         if not isinstance(training_metrics, dict):
 229 |             return {}
 230 |         
 231 |         entropy_data = training_metrics.get('policy_entropy_history', []) 
 232 |         if entropy_data:
 233 |             enhanced_dynamics['policy_entropy'] = self._analyze_policy_entropy(entropy_data)
 234 |         
 235 |         # --- Training stability analysis ---
 236 |         reward_data = training_metrics.get('cumulative_reward_history', [])
 237 |         if reward_data:
 238 |             enhanced_dynamics['training_stability'] = self._analyze_training_stability(reward_data)
 239 |         
 240 |         # --- Loss analysis ---
 241 |         for loss_type in ['policy_loss', 'value_loss', 'distillation_loss']:
 242 |             # CORRECTED: Access the '_history' list for each loss type
 243 |             loss_data = training_metrics.get(f'{loss_type}_history', [])
 244 |             if loss_data:
 245 |                 enhanced_dynamics[f'{loss_type}_analysis'] = self._analyze_loss_trajectory(loss_data)
 246 |         
 247 |         # --- Convergence analysis ---
 248 |         # This analysis needs the policy loss history to function correctly.
 249 |         policy_loss_data = training_metrics.get('policy_loss_history', [])
 250 |         if policy_loss_data:
 251 |             # Pass the specific data it needs instead of the whole dictionary
 252 |             enhanced_dynamics['convergence_metrics'] = self._analyze_convergence(policy_loss_data)
 253 | 
 254 |         # --- Method-specific analysis (This logic is preserved) ---
 255 |         if method_name == 'rl_ttm':
 256 |             enhanced_dynamics['ttm_specific'] = self._analyze_ttm_specific_metrics(training_metrics)
 257 |         elif method_name == 'offline_pkt':
 258 |             enhanced_dynamics['pkt_specific'] = self._analyze_pkt_specific_metrics(training_metrics)
 259 |         
 260 |         return enhanced_dynamics
 261 |     def _calculate_behavioral_metrics(self, eval_dict: Dict) -> Dict[str, float]:
 262 |         """Calculate behavioral and trading statistics."""
 263 |         all_positions = []
 264 |         trade_counts = []
 265 |         
 266 |         for pair_name, pair_data in eval_dict.items():
 267 |             if isinstance(pair_data, dict) and 'position' in pair_data:
 268 |                 positions = pair_data['position']
 269 |                 all_positions.extend(positions)
 270 |                 
 271 |                 # Count position changes as trades
 272 |                 position_changes = np.diff(positions)
 273 |                 trades = np.sum(position_changes != 0)
 274 |                 trade_counts.append(trades)
 275 |         
 276 |         if not all_positions:
 277 |             return {}
 278 |         
 279 |         # Calculate metrics
 280 |         turnover_rate = np.mean(trade_counts) if trade_counts else 0
 281 |         position_concentration = self._calculate_position_concentration(all_positions)
 282 |         
 283 |         return {
 284 |             'turnover_rate': turnover_rate,
 285 |             'avg_position_size': np.mean(np.abs(all_positions)),
 286 |             'position_concentration': position_concentration,
 287 |             'max_position': np.max(np.abs(all_positions)),
 288 |             'position_std': np.std(all_positions),
 289 |             'long_short_ratio': self._calculate_long_short_ratio(all_positions)
 290 |         }
 291 |     
 292 |     def _extract_baseline_pnl(self, all_results: Dict[str, Any]) -> Optional[pd.Series]:
 293 |         """
 294 |         Extract baseline PnL curve from no_distillation results.
 295 |         
 296 |         Args:
 297 |             all_results: Dictionary of all method results from orchestrator
 298 |             
 299 |         Returns:
 300 |             Average baseline PnL curve or None if not available
 301 |         """
 302 |         baseline_method = 'no_distillation'
 303 |         
 304 |         if baseline_method not in all_results:
 305 |             self.logger.warning(f"Baseline method '{baseline_method}' not found in results")
 306 |             return None
 307 |             
 308 |         baseline_data = all_results[baseline_method]
 309 |         if 'results' not in baseline_data:
 310 |             self.logger.warning(f"No results found for baseline method '{baseline_method}'")
 311 |             return None
 312 |             
 313 |         baseline_results = baseline_data['results']
 314 |         
 315 |         # Extract PnL curves from all runs
 316 |         pnl_curves = []
 317 |         if 'runs' in baseline_results:
 318 |             for run in baseline_results['runs']:
 319 |                 if run.get('avg_test_pnl') is not None:
 320 |                     pnl_curves.append(run['avg_test_pnl'])
 321 |         
 322 |         if not pnl_curves:
 323 |             self.logger.warning("No PnL curves found in baseline results")
 324 |             return None
 325 |             
 326 |         # Calculate average baseline PnL curve
 327 |         try:
 328 |             avg_baseline_pnl = pd.concat(pnl_curves, axis=1).mean(axis=1)
 329 |             self.logger.info("✓ Successfully extracted average baseline PnL curve")
 330 |             return avg_baseline_pnl
 331 |         except Exception as e:
 332 |             self.logger.error(f"Failed to calculate average baseline PnL: {e}")
 333 |             return None
 334 |     
 335 |     def generate_academic_report(self, all_results: Dict[str, Any]) -> Dict[str, Path]:
 336 | 
 337 |         
 338 |         # 1. Extract baseline PnL for comparison plots
 339 |         avg_baseline_pnl = self._extract_baseline_pnl(all_results)
 340 |         
 341 |         # 2. Perform all analysis and enhancement
 342 |         enhanced_results = {}
 343 |         for method_name, method_data in all_results.items():
 344 |             if 'results' in method_data:
 345 |                 enhanced_results[method_name] = self.enhance_experiment_results(
 346 |                     method_name, method_data['results']
 347 |                 )
 348 |                 
 349 |         # 2. Hand off the final, enhanced data to the evaluator for reporting
 350 |         methods = list(enhanced_results.keys())
 351 |         output_files = self.evaluator.generate_comparison_report(
 352 |             methods=methods,
 353 |             all_enhanced_data=enhanced_results,
 354 |             output_format='both',
 355 |             avg_baseline_pnl=avg_baseline_pnl
 356 |         )
 357 |         
 358 |         return output_files
 359 |     
 360 |     def _generate_additional_artifacts(self, enhanced_results: Dict) -> Dict[str, Path]:
 361 |         """Generate additional academic artifacts like detailed data exports."""
 362 |         artifacts = {}
 363 |         
 364 |         # Export detailed metrics CSV
 365 |         detailed_metrics_path = self.academic_output_dir / "detailed_metrics_export.csv"
 366 |         self._export_detailed_metrics_csv(enhanced_results, detailed_metrics_path)
 367 |         artifacts['detailed_metrics_csv'] = detailed_metrics_path
 368 |         
 369 |         # Generate method comparison matrix
 370 |         comparison_matrix_path = self.academic_output_dir / "method_comparison_matrix.csv"
 371 |         self._generate_comparison_matrix(enhanced_results, comparison_matrix_path)
 372 |         artifacts['comparison_matrix'] = comparison_matrix_path
 373 |     
 374 |     def _extract_method_training_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
 375 |         """
 376 |         Extract training metrics from method-level data (for academic approach).
 377 |         
 378 |         Args:
 379 |             results: Raw experiment results from orchestrator
 380 |             
 381 |         Returns:
 382 |             Dictionary containing training metrics in the expected format
 383 |         """
 384 |         # Check if training metrics are available at the top level (academic approach)
 385 |         training_metrics = {}
 386 |         
 387 |         # Look for training metrics in various possible locations
 388 |         possible_keys = [
 389 |             'final_train_metrics',
 390 |             'policy_loss_history',
 391 |             'value_loss_history',
 392 |             'distillation_loss_history',
 393 |             'policy_entropy_history',
 394 |             'cumulative_reward_history',
 395 |             'model_stats',
 396 |             'final_lr',
 397 |             'total_epochs'
 398 |         ]
 399 |         
 400 |         for key in possible_keys:
 401 |             if key in results:
 402 |                 training_metrics[key] = results[key]
 403 |         
 404 |         # If no direct training metrics found, check if they're nested somewhere
 405 |         if not training_metrics:
 406 |             # Check in aggregates or other nested structures
 407 |             aggregates = results.get('aggregates', {})
 408 |             if aggregates:
 409 |                 for key in possible_keys:
 410 |                     if key in aggregates:
 411 |                         training_metrics[key] = aggregates[key]
 412 |         
 413 |         return training_metrics if training_metrics else None
 414 |     
 415 |     def _create_empty_financial_metrics(self) -> Dict[str, float]:
 416 |         """Create empty financial metrics structure to prevent KeyErrors."""
 417 |         return {
 418 |             'cumulative_return': 0.0,
 419 |             'total_return': 0.0,
 420 |             'sharpe_ratio': 0.0,
 421 |             'sortino_ratio': 0.0,
 422 |             'calmar_ratio': 0.0,
 423 |             'volatility': 0.0,
 424 |             'max_drawdown': 0.0,
 425 |             'profit_factor': 0.0
 426 |         }
 427 |     
 428 |     def _extract_method_financial_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
 429 |         """
 430 |         Extract financial data from method-level data (for academic approach).
 431 |         
 432 |         Args:
 433 |             results: Raw experiment results from orchestrator
 434 |             
 435 |         Returns:
 436 |             Dictionary containing financial data in the expected format
 437 |         """
 438 |         financial_data = {}
 439 |         
 440 |         # Check for PnL data in results structure
 441 |         if 'pnl_curves' in results:
 442 |             financial_data['pnl_curves'] = results['pnl_curves']
 443 |         
 444 |         # Check for aggregated performance data
 445 |         aggregates = results.get('aggregates', {})
 446 |         if aggregates:
 447 |             performance = aggregates.get('performance', {})
 448 |             if performance:
 449 |                 # Extract the actual PnL curve data
 450 |                 if 'pnl_mean_curve' in performance:
 451 |                     financial_data['pnl_mean_curve'] = performance['pnl_mean_curve']
 452 |                 
 453 |                 # Extract other financial metrics
 454 |                 for key in ['annualized_pnl_mean', 'cumulative_pnl_mean', 'sharpe_ratio', 'max_drawdown']:
 455 |                     if key in performance:
 456 |                         financial_data[key] = performance[key]
 457 |         
 458 |         return financial_data if financial_data else None
 459 |     
 460 |     def _create_empty_behavioral_metrics(self) -> Dict[str, float]:
 461 |         """Create empty behavioral metrics structure to prevent aggregation issues."""
 462 |         return {
 463 |             'turnover_rate': 0.0,
 464 |             'avg_position_size': 0.0,
 465 |             'position_concentration': 0.0,
 466 |             'max_position': 0.0,
 467 |             'position_std': 0.0,
 468 |             'long_short_ratio': 0.0,
 469 |             'trade_frequency': 0.0,
 470 |             'avg_holding_period': 0.0,
 471 |             'var_95': 0.0
 472 |         }
 473 |         
 474 |         # Export training dynamics data
 475 |         dynamics_path = self.academic_output_dir / "training_dynamics_export.csv"
 476 |         self._export_training_dynamics(enhanced_results, dynamics_path)
 477 |         artifacts['training_dynamics'] = dynamics_path
 478 |         
 479 |         return artifacts
 480 |     
 481 |     def _calculate_method_academic_metrics(self, method_name: str, runs: List[Dict]) -> Dict[str, Any]:
 482 |         """Calculate method-level academic metrics across all runs."""
 483 |         if not runs:
 484 |             return {}
 485 |         
 486 |         # Aggregate financial metrics
 487 |         financial_metrics = [run.get('financial_metrics', {}) for run in runs if run.get('financial_metrics')]
 488 |         financial_aggregates = self._aggregate_metrics(financial_metrics)
 489 |         
 490 |         # Aggregate risk metrics
 491 |         risk_metrics = [run.get('risk_metrics', {}) for run in runs if run.get('risk_metrics')]
 492 |         risk_aggregates = self._aggregate_metrics(risk_metrics)
 493 |         
 494 |         # Aggregate behavioral metrics
 495 |         behavioral_metrics = [run.get('behavioral_metrics', {}) for run in runs if run.get('behavioral_metrics')]
 496 |         behavioral_aggregates = self._aggregate_metrics(behavioral_metrics)
 497 |         
 498 |         return {
 499 |             'financial': financial_aggregates,
 500 |             'risk': risk_aggregates,
 501 |             'behavioral': behavioral_aggregates,
 502 |             'n_successful_runs': len([r for r in runs if 'error' not in r]),
 503 |             'method_name': method_name
 504 |         }
 505 |     
 506 |     def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict[str, Dict[str, float]]:
 507 |         """Aggregate metrics across runs with mean and std."""
 508 |         if not metrics_list:
 509 |             return {}
 510 |         
 511 |         aggregated = {}
 512 |         all_keys = set()
 513 |         for metrics in metrics_list:
 514 |             all_keys.update(metrics.keys())
 515 |         
 516 |         for key in all_keys:
 517 |             values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
 518 |             if values:
 519 |                 aggregated[key] = {
 520 |                     'mean': np.mean(values),
 521 |                     'std': np.std(values),
 522 |                     'min': np.min(values),
 523 |                     'max': np.max(values),
 524 |                     'count': len(values)
 525 |                 }
 526 |         
 527 |         return aggregated
 528 |     
 529 |     # Helper methods for specific calculations
 530 |     def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
 531 |         """Calculate annualized Sharpe ratio."""
 532 |         if len(returns) == 0 or returns.std() == 0:
 533 |             return 0
 534 |         return (returns.mean() / returns.std()) * np.sqrt(252)
 535 |     
 536 |     def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
 537 |         """Calculate Sortino ratio (downside deviation only)."""
 538 |         if len(returns) == 0:
 539 |             return 0
 540 |         downside_returns = returns[returns < 0]
 541 |         if len(downside_returns) == 0:
 542 |             return float('inf')
 543 |         downside_std = downside_returns.std()
 544 |         if downside_std == 0:
 545 |             return 0
 546 |         return (returns.mean() / downside_std) * np.sqrt(252)
 547 |     
 548 |     def _calculate_calmar_ratio(self, pnl_series: pd.Series) -> float:
 549 |         """Calculate Calmar ratio (return / max drawdown)."""
 550 |         if len(pnl_series) < 2:
 551 |             return 0
 552 |         
 553 |         total_return = (pnl_series.iloc[-1] / pnl_series.iloc[0] - 1) if pnl_series.iloc[0] != 0 else 0
 554 |         
 555 |         # Calculate max drawdown
 556 |         cumulative = (1 + pnl_series.pct_change().fillna(0)).cumprod()
 557 |         running_max = cumulative.expanding().max()
 558 |         drawdown = (cumulative - running_max) / running_max
 559 |         max_drawdown = abs(drawdown.min())
 560 |         
 561 |         return total_return / max_drawdown if max_drawdown > 0 else 0
 562 |     
 563 |     def _calculate_drawdown_duration(self, drawdown_series: pd.Series) -> float:
 564 |         """Calculate average drawdown duration."""
 565 |         in_drawdown = drawdown_series < 0
 566 |         if not in_drawdown.any():
 567 |             return 0
 568 |         
 569 |         # Find drawdown periods
 570 |         drawdown_periods = []
 571 |         current_period = 0
 572 |         
 573 |         for is_dd in in_drawdown:
 574 |             if is_dd:
 575 |                 current_period += 1
 576 |             else:
 577 |                 if current_period > 0:
 578 |                     drawdown_periods.append(current_period)
 579 |                 current_period = 0
 580 |         
 581 |         if current_period > 0:
 582 |             drawdown_periods.append(current_period)
 583 |         
 584 |         return np.mean(drawdown_periods) if drawdown_periods else 0
 585 |     
 586 |     def _analyze_policy_entropy(self, entropy_data: List[float]) -> Dict[str, float]:
 587 |         """Analyze policy entropy trends - crucial for TTM evaluation."""
 588 |         if not entropy_data:
 589 |             return {}
 590 |         
 591 |         entropy_array = np.array(entropy_data)
 592 |         
 593 |         # Trend analysis
 594 |         if len(entropy_array) > 10:
 595 |             early_entropy = np.mean(entropy_array[:len(entropy_array)//4])
 596 |             late_entropy = np.mean(entropy_array[-len(entropy_array)//4:])
 597 |             trend_ratio = (late_entropy - early_entropy) / early_entropy if early_entropy > 0 else 0
 598 |         else:
 599 |             trend_ratio = 0
 600 |         
 601 |         return {
 602 |             'mean_entropy': np.mean(entropy_array),
 603 |             'std_entropy': np.std(entropy_array),
 604 |             'min_entropy': np.min(entropy_array),
 605 |             'max_entropy': np.max(entropy_array),
 606 |             'trend_ratio': trend_ratio,
 607 |             'final_entropy': entropy_array[-1] if len(entropy_array) > 0 else 0
 608 |         }
 609 |     
 610 |     def _analyze_training_stability(self, reward_data: List[float]) -> Dict[str, float]:
 611 |         """Analyze training stability from reward progression."""
 612 |         if not reward_data:
 613 |             return {}
 614 |         
 615 |         reward_array = np.array(reward_data)
 616 |         
 617 |         # Calculate stability metrics
 618 |         if len(reward_array) > 10:
 619 |             # Coefficient of variation in later training
 620 |             late_rewards = reward_array[-len(reward_array)//4:]
 621 |             cv = np.std(late_rewards) / np.mean(late_rewards) if np.mean(late_rewards) != 0 else float('inf')
 622 |             
 623 |             # Trend strength
 624 |             x = np.arange(len(reward_array))
 625 |             trend_slope = np.polyfit(x, reward_array, 1)[0] if len(reward_array) > 1 else 0
 626 |         else:
 627 |             cv = float('inf')
 628 |             trend_slope = 0
 629 |         
 630 |         return {
 631 |             'coefficient_of_variation': cv,
 632 |             'trend_slope': trend_slope,
 633 |             'final_reward': reward_array[-1] if len(reward_array) > 0 else 0,
 634 |             'reward_range': np.max(reward_array) - np.min(reward_array),
 635 |             'stability_score': 1.0 / (cv + 1e-8)  # Higher is more stable
 636 |         }
 637 |     
 638 |     def _analyze_loss_trajectory(self, loss_data: List[float]) -> Dict[str, float]:
 639 |         """Analyze loss trajectory for convergence insights."""
 640 |         if not loss_data:
 641 |             return {}
 642 |         
 643 |         loss_array = np.array(loss_data)
 644 |         
 645 |         return {
 646 |             'initial_loss': loss_array[0] if len(loss_array) > 0 else 0,
 647 |             'final_loss': loss_array[-1] if len(loss_array) > 0 else 0,
 648 |             'min_loss': np.min(loss_array),
 649 |             'loss_reduction': (loss_array[0] - loss_array[-1]) / loss_array[0] if len(loss_array) > 0 and loss_array[0] != 0 else 0,
 650 |             'loss_volatility': np.std(loss_array)
 651 |         }
 652 |     
 653 |     def _analyze_convergence(self, training_metrics: Dict) -> Dict[str, Any]:
 654 |         """Analyze convergence characteristics."""
 655 |         convergence_metrics = {}
 656 |         
 657 |         # Look for signs of convergence in various metrics
 658 |         for metric_name in ['reward', 'policy_loss', 'value_loss']:
 659 |             if metric_name in training_metrics:
 660 |                 data = training_metrics[metric_name]
 661 |                 if isinstance(data, list) and len(data) > 20:
 662 |                     # Simple convergence detection: variance in last 20% of training
 663 |                     late_portion = data[-len(data)//5:]
 664 |                     convergence_metrics[f'{metric_name}_converged'] = np.std(late_portion) < 0.01
 665 |         
 666 |         return convergence_metrics
 667 |     
 668 | # In AcademicIntegration class
 669 | 
 670 |     def _analyze_ttm_specific_metrics(self, training_metrics: Dict) -> Dict[str, Any]:
 671 |         """
 672 |         Analyzes TTM-specific metrics, focusing on distillation effectiveness
 673 |         and the resulting policy regularization.
 674 |         """
 675 |         ttm_metrics = {}
 676 |         
 677 |         # --- 1. Distillation Effectiveness (Your original logic, which is good) ---
 678 |         distillation_history = training_metrics.get('distillation_loss_history', [])
 679 |         if distillation_history:
 680 |             # Filter out any potential non-numeric values or NaNs before calculation
 681 |             valid_losses = [loss for loss in distillation_history if isinstance(loss, (int, float)) and np.isfinite(loss)]
 682 |             if len(valid_losses) > 1:
 683 |                 initial_loss = valid_losses[0]
 684 |                 final_loss = valid_losses[-1]
 685 |                 ttm_metrics['distillation_effectiveness'] = {
 686 |                     'initial_loss': initial_loss,
 687 |                     'final_loss': final_loss,
 688 |                     'reduction_ratio': (initial_loss - final_loss) / initial_loss if initial_loss != 0 else 0,
 689 |                     'min_loss_achieved': min(valid_losses)
 690 |                 }
 691 | 
 692 |         # --- 2. Policy Regularization Analysis (The key addition) ---
 693 |         # This directly measures the theoretical goal of TTM.
 694 |         entropy_history = training_metrics.get('policy_entropy_history', [])
 695 |         if len(entropy_history) > 1:
 696 |             ttm_metrics['policy_regularization'] = {
 697 |                 'mean_policy_entropy': np.mean(entropy_history),
 698 |                 'final_policy_entropy': entropy_history[-1],
 699 |                 # Coefficient of Variation: Lower is more stable entropy
 700 |                 'entropy_stability_cv': np.std(entropy_history) / np.mean(entropy_history) if np.mean(entropy_history) != 0 else 0,
 701 |             }
 702 |             
 703 |         return ttm_metrics
 704 |         
 705 |     # In AcademicIntegration class
 706 | 
 707 |     def _analyze_pkt_specific_metrics(self, training_metrics: Dict) -> Dict[str, Any]:
 708 |         """
 709 |         Analyzes PKT-specific metrics, focusing on the convergence of the
 710 |         feature-space distribution matching.
 711 |         """
 712 |         pkt_metrics = {}
 713 |         
 714 |         # --- 1. Feature Distribution Alignment Analysis ---
 715 |         # The distillation_loss for PKT is the L_pkt (KL divergence).
 716 |         # We analyze how well this loss was minimized.
 717 |         pkt_loss_history = training_metrics.get('distillation_loss_history', [])
 718 |         if pkt_loss_history:
 719 |             valid_losses = [loss for loss in pkt_loss_history if isinstance(loss, (int, float)) and np.isfinite(loss)]
 720 |             if len(valid_losses) > 1:
 721 |                 initial_loss = valid_losses[0]
 722 |                 final_loss = valid_losses[-1]
 723 |                 pkt_metrics['distribution_alignment'] = {
 724 |                     'initial_pkt_loss': initial_loss,
 725 |                     'final_pkt_loss': final_loss,
 726 |                     'loss_reduction_ratio': (initial_loss - final_loss) / initial_loss if initial_loss != 0 else 0,
 727 |                     'min_pkt_loss_achieved': min(valid_losses),
 728 |                     'loss_convergence_stability': np.std(valid_losses[-20:]) if len(valid_losses) > 20 else np.std(valid_losses)
 729 |                 }
 730 |                 
 731 |         # --- 2. Impact on Policy (Indirect measure) ---
 732 |         # We can also check if aligning the features had a positive effect on the policy's stability.
 733 |         entropy_history = training_metrics.get('policy_entropy_history', [])
 734 |         if len(entropy_history) > 1:
 735 |             pkt_metrics['policy_impact'] = {
 736 |                 'mean_policy_entropy': np.mean(entropy_history),
 737 |                 'entropy_stability_cv': np.std(entropy_history) / np.mean(entropy_history) if np.mean(entropy_history) != 0 else 0,
 738 |             }
 739 |             
 740 |         return pkt_metrics
 741 |     
 742 |     def _calculate_position_concentration(self, positions: List[float]) -> float:
 743 |         """Calculate position concentration using Herfindahl index."""
 744 |         if not positions:
 745 |             return 0
 746 |         
 747 |         abs_positions = np.abs(positions)
 748 |         total_exposure = np.sum(abs_positions)
 749 |         
 750 |         if total_exposure == 0:
 751 |             return 0
 752 |         
 753 |         weights = abs_positions / total_exposure
 754 |         return np.sum(weights ** 2)
 755 |     
 756 |     def _calculate_long_short_ratio(self, positions: List[float]) -> float:
 757 |         """Calculate ratio of long to short positions."""
 758 |         if not positions:
 759 |             return 1.0
 760 |         
 761 |         long_positions = [p for p in positions if p > 0]
 762 |         short_positions = [p for p in positions if p < 0]
 763 |         
 764 |         long_sum = sum(long_positions)
 765 |         short_sum = abs(sum(short_positions))
 766 |         
 767 |         if short_sum == 0:
 768 |             return float('inf') if long_sum > 0 else 1.0
 769 |         
 770 |         return long_sum / short_sum
 771 |     
 772 |     def _export_detailed_metrics_csv(self, enhanced_results: Dict, output_path: Path):
 773 |         """Export detailed metrics to CSV for further analysis."""
 774 |         rows = []
 775 |         
 776 |         for method_name, method_data in enhanced_results.items():
 777 |             for run_idx, run in enumerate(method_data.get('runs', [])):
 778 |                 row = {
 779 |                     'method': method_name,
 780 |                     'run_idx': run_idx,
 781 |                     'timestamp': run.get('timestamp', ''),
 782 |                     'unique_id': run.get('unique_id', '')
 783 |                 }
 784 |                 
 785 |                 # Add financial metrics
 786 |                 financial = run.get('financial_metrics', {})
 787 |                 for key, value in financial.items():
 788 |                     row[f'financial_{key}'] = value
 789 |                 
 790 |                 # Add risk metrics
 791 |                 risk = run.get('risk_metrics', {})
 792 |                 for key, value in risk.items():
 793 |                     row[f'risk_{key}'] = value
 794 |                 
 795 |                 # Add behavioral metrics
 796 |                 behavioral = run.get('behavioral_metrics', {})
 797 |                 for key, value in behavioral.items():
 798 |                     row[f'behavioral_{key}'] = value
 799 |                 
 800 |                 rows.append(row)
 801 |         
 802 |         if rows:
 803 |             df = pd.DataFrame(rows)
 804 |             df.to_csv(output_path, index=False)
 805 |             self.logger.info(f"Detailed metrics exported to: {output_path}")
 806 |     
 807 |     def _generate_comparison_matrix(self, enhanced_results: Dict, output_path: Path):
 808 |         """Generate method comparison matrix."""
 809 |         methods = list(enhanced_results.keys())
 810 |         metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown', 'volatility']
 811 |         
 812 |         comparison_data = []
 813 |         for method in methods:
 814 |             row = {'method': method}
 815 |             runs = enhanced_results[method].get('runs', [])
 816 |             
 817 |             for metric in metrics:
 818 |                 values = []
 819 |                 for run in runs:
 820 |                     financial = run.get('financial_metrics', {})
 821 |                     risk = run.get('risk_metrics', {})
 822 |                     
 823 |                     if metric in financial:
 824 |                         values.append(financial[metric])
 825 |                     elif metric in risk:
 826 |                         values.append(risk[metric])
 827 |                 
 828 |                 if values:
 829 |                     row[f'{metric}_mean'] = np.mean(values)
 830 |                     row[f'{metric}_std'] = np.std(values)
 831 |                 else:
 832 |                     row[f'{metric}_mean'] = 0
 833 |                     row[f'{metric}_std'] = 0
 834 |             
 835 |             comparison_data.append(row)
 836 |         
 837 |         if comparison_data:
 838 |             df = pd.DataFrame(comparison_data)
 839 |             df.to_csv(output_path, index=False)
 840 |             self.logger.info(f"Comparison matrix exported to: {output_path}")
 841 |     
 842 |     def _export_training_dynamics(self, enhanced_results: Dict, output_path: Path):
 843 |         """Export training dynamics data for detailed analysis."""
 844 |         dynamics_data = []
 845 |         
 846 |         for method_name, method_data in enhanced_results.items():
 847 |             for run_idx, run in enumerate(method_data.get('runs', [])):
 848 |                 rl_dynamics = run.get('rl_dynamics', {})
 849 |                 
 850 |                 for metric_name, metric_data in rl_dynamics.items():
 851 |                     if isinstance(metric_data, dict):
 852 |                         row = {
 853 |                             'method': method_name,
 854 |                             'run_idx': run_idx,
 855 |                             'metric_category': metric_name
 856 |                         }
 857 |                         row.update(metric_data)
 858 |                         dynamics_data.append(row)
 859 |         
 860 |         if dynamics_data:
 861 |             df = pd.DataFrame(dynamics_data)
 862 |             df.to_csv(output_path, index=False)
 863 |             self.logger.info(f"Training dynamics exported to: {output_path}")
Project Path: academic_evaluator.py

Source Tree:

```
academic_evaluator.py

```

`\\?\C:\Users\George\Source\Github\Thesis\financial-trading-in-RL\experiment_code\academic_evaluator.py`:

```py
   1 | """
   2 | Academic Evaluator for TTM vs PKT Comparison
   3 | ============================================
   4 | 
   5 | This module implements the comprehensive evaluation strategy dictated by the Master Engineer
   6 | for rigorous, multi-faceted evaluation of TTM and PKT distillation methods.
   7 | 
   8 | The evaluation captures performance from three distinct perspectives:
   9 | 1. Financial Performance: Risk-adjusted returns (ground truth)
  10 | 2. Reinforcement Learning Dynamics: Learning process analysis
  11 | 3. Behavioral Statistics: Trading strategy characterization
  12 | 
  13 | All results are reported with mean ± standard deviation across multiple independent runs.
  14 | """
  15 | 
  16 | import numpy as np
  17 | import pandas as pd
  18 | from typing import Dict, List, Any, Optional, Tuple
  19 | from dataclasses import dataclass
  20 | from pathlib import Path
  21 | import matplotlib.pyplot as plt
  22 | import seaborn as sns
  23 | from datetime import datetime
  24 | 
  25 | from experiment_code.core.report_engine import ReportEngine
  26 | from experiment_code.core.formatters.latex_formatter import LaTeXFormatter
  27 | from experiment_code.core.formatters.markdown_formatter import MarkdownFormatter
  28 | from experiment_code.core.data_extractor import DataExtractor
  29 | from experiment_code.core.table_generator import TableGenerator
  30 | 
  31 | # Import visualization functions from the centralized module
  32 | from experiment_code.report_visualizations import (
  33 |     generate_training_progress,
  34 |     generate_pnl_over_time,
  35 |     generate_relative_performance,
  36 |     create_thesis_style_plots,
  37 |     create_thesis_style_comparison_plots,
  38 |     create_trading_analysis_plots
  39 | )
  40 | 
  41 | 
  42 | @dataclass
  43 | class FinancialMetrics:
  44 |     """Financial Performance Metrics - The ultimate 'ground truth'"""
  45 |     cumulative_pnl: float
  46 |     sharpe_ratio: float
  47 |     sortino_ratio: float
  48 |     max_drawdown: float
  49 |     calmar_ratio: float
  50 |     profit_factor: float
  51 |     
  52 |     # Statistical measures
  53 |     mean: float
  54 |     std: float
  55 |     
  56 |     @property
  57 |     def formatted_result(self) -> str:
  58 |         return f"{self.mean:.3f} ± {self.std:.3f}"
  59 | 
  60 | 
  61 | @dataclass
  62 | class RLDynamicsMetrics:
  63 |     """Reinforcement Learning & Training Dynamics"""
  64 |     cumulative_reward: List[float]
  65 |     policy_entropy: List[float]
  66 |     policy_loss: List[float]
  67 |     value_loss: List[float]
  68 |     distillation_loss: Optional[List[float]]
  69 |     
  70 |     # Derived metrics
  71 |     entropy_mean: float
  72 |     entropy_std: float
  73 |     convergence_epoch: Optional[int]
  74 |     training_stability: float
  75 |     
  76 |     @property
  77 |     def entropy_trend(self) -> str:
  78 |         """Analyze entropy trend - key diagnostic for TTM"""
  79 |         if len(self.policy_entropy) < 20:
  80 |             return "insufficient_data"
  81 |         
  82 |         early_window = np.mean(self.policy_entropy[:20])
  83 |         late_window = np.mean(self.policy_entropy[-20:])
  84 |         trend_ratio = (late_window - early_window) / early_window
  85 |         
  86 |         if trend_ratio > 0.1:
  87 |             return "increasing"
  88 |         elif trend_ratio < -0.1:
  89 |             return "decreasing"
  90 |         else:
  91 |             return "stable"
  92 | 
  93 | 
  94 | @dataclass
  95 | class BehavioralMetrics:
  96 |     """Behavioral & Trading Statistics"""
  97 |     turnover_rate: float
  98 |     avg_holding_period: float
  99 |     win_rate: float
 100 |     trade_frequency: float
 101 |     position_concentration: float
 102 |     
 103 |     # Risk metrics
 104 |     var_95: float
 105 |     expected_shortfall: float
 106 | 
 107 | 
 108 | class AcademicEvaluator:
 109 |     """
 110 |     Comprehensive academic evaluation system for TTM vs PKT comparison.
 111 |     
 112 |     Implements the Master Engineer's evaluation strategy with rigorous
 113 |     statistical analysis and academic-quality reporting.
 114 |     """
 115 |     
 116 |     def __init__(self, output_dir: Path):
 117 |         self.output_dir = Path(output_dir)
 118 |         self.output_dir.mkdir(parents=True, exist_ok=True)
 119 |         
 120 |         # Initialize report engine with both formatters
 121 |         self.report_engine = ReportEngine()
 122 |         self.report_engine.register_formatter('latex', LaTeXFormatter())
 123 |         self.report_engine.register_formatter('markdown', MarkdownFormatter())
 124 |         
 125 |         # Set up data extractor and table generator
 126 |         self.report_engine.set_data_extractor(DataExtractor())
 127 |         self.report_engine.set_table_generator(TableGenerator())
 128 |         
 129 |         # Store results for comparison
 130 |         self.method_results: Dict[str, Dict] = {}
 131 |         
 132 |     def store_method_results(self, method_name: str, enhanced_results: Dict[str, Any]) -> None:
 133 |         """
 134 |         Store enhanced results from AcademicIntegration for reporting.
 135 |         
 136 |         Args:
 137 |             method_name: Name of the method ('rl_ttm', 'offline_pkt', 'no_distillation')
 138 |             enhanced_results: Enhanced results with academic metrics from AcademicIntegration
 139 |         """
 140 |         print(f"\n=== Storing enhanced results for {method_name} ===")
 141 |         
 142 |         # Store the enhanced results directly - analysis is already done by AcademicIntegration
 143 |         self.method_results[method_name] = enhanced_results
 144 |     
 145 |     # All calculation methods removed - analysis is now handled by AcademicIntegration
 146 |     # This class is now a pure reporting engine
 147 |     
 148 | # Replace the existing generate_comparison_report method in AcademicEvaluator with this one.
 149 | 
 150 |     def generate_comparison_report(self, methods: List[str], all_enhanced_data: Dict[str, Any], output_format: str = 'both', avg_baseline_pnl: Optional[pd.Series] = None) -> Dict[str, Path]:
 151 |         """
 152 |         Generate comprehensive comparison report as a pure reporting engine.
 153 |         Now receives fully analyzed data from AcademicIntegration.
 154 |         
 155 |         Args:
 156 |             methods: List of method names
 157 |             all_enhanced_data: Fully enhanced data from AcademicIntegration
 158 |             output_format: Output format ('latex', 'markdown', or 'both')
 159 |             avg_baseline_pnl: Average baseline PnL curve for comparison plots
 160 |             
 161 |         Returns:
 162 |             Dictionary mapping report types to file paths
 163 |         """
 164 |         print(f"\n=== Formatting Final Academic Report ===")
 165 |         
 166 |         # Store the enhanced data
 167 |         self.method_results = all_enhanced_data
 168 |         
 169 |         # If baseline PnL is provided, add it to method_results for visualization functions
 170 |         if avg_baseline_pnl is not None:
 171 |             # Create a baseline entry that matches the expected structure for visualizations
 172 |             baseline_entry = {
 173 |                 'results': {
 174 |                     'runs': [{'avg_test_pnl': avg_baseline_pnl}]
 175 |                 },
 176 |                 'aggregates': {
 177 |                     'performance': {
 178 |                         'pnl_mean_curve': avg_baseline_pnl
 179 |                     }
 180 |                 }
 181 |             }
 182 |             self.method_results['no_distillation'] = baseline_entry
 183 |             print("✓ Added baseline PnL data to method_results for visualization")
 184 |         
 185 |         output_files = {}
 186 |         
 187 |         # --- 1. Generate Text/Table Reports (LaTeX, Markdown, CSV) ---
 188 |         comparison_data = {
 189 |             'title': 'Academic Comparison Report: Distillation Methods',
 190 |             'metadata': {
 191 |                 'methods': methods,
 192 |                 'timestamp': datetime.now().isoformat(),
 193 |             },
 194 |             'methods_overview': {
 195 |                 'methods': [self.method_results[m] for m in methods if m in self.method_results]
 196 |             },
 197 |             'performance_comparison': {
 198 |                 'summary_table': self._create_summary_table(methods)
 199 |             },
 200 |             'conclusions': {
 201 |                 'text': self._generate_conclusions(methods)
 202 |             }
 203 |         }
 204 |         
 205 |         if output_format in ['latex', 'both']:
 206 |             latex_file = self.output_dir / 'academic_evaluation_report.tex'
 207 |             try:
 208 |                 self.report_engine.generate_report(
 209 |                     template_type='comparison',
 210 |                     format_type='latex',
 211 |                     data=comparison_data,
 212 |                     output_path=latex_file
 213 |                 )
 214 |                 output_files['latex_report'] = latex_file
 215 |                 print(f"✓ LaTeX report generated: {latex_file}")
 216 |             except Exception as e:
 217 |                 print(f"⚠️ Failed to generate LaTeX report: {e}")
 218 | 
 219 |         if output_format in ['markdown', 'both']:
 220 |             md_file = self.output_dir / 'academic_evaluation_report.md'
 221 |             try:
 222 |                 self.report_engine.generate_report(
 223 |                     template_type='comparison',
 224 |                     format_type='markdown',
 225 |                     data=comparison_data,
 226 |                     output_path=md_file
 227 |                 )
 228 |                 output_files['markdown_report'] = md_file
 229 |                 print(f"✓ Markdown report generated: {md_file}")
 230 |             except Exception as e:
 231 |                 print(f"⚠️ Failed to generate Markdown report: {e}")
 232 | 
 233 |         # Generate summary table CSV
 234 |         csv_file = self.output_dir / 'summary_comparison_table.csv'
 235 |         self._save_summary_csv(methods, csv_file)
 236 |         output_files['summary_csv'] = csv_file
 237 |         
 238 |         # --- 2. Generate Visualizations by calling the utility functions ---
 239 |         print("  > Generating key visualizations...")
 240 |         
 241 |         try:
 242 |             # Training progress plot
 243 |             training_plot_path = self.output_dir / 'training_progress.html'
 244 |             generate_training_progress(self.method_results, training_plot_path, "Agent's Performance for All Types of Distillation")
 245 |             output_files['training_progress'] = training_plot_path
 246 |         except Exception as e:
 247 |             print(f"⚠️ Failed to generate training progress plot: {e}")
 248 |             
 249 |         try:
 250 |             # PnL over time plot
 251 |             pnl_plot_path = self.output_dir / 'pnl_over_time.html'
 252 |             generate_pnl_over_time(self.method_results, pnl_plot_path, "Agent's Performance Comparison Over Time")
 253 |             output_files['pnl_over_time'] = pnl_plot_path
 254 |         except Exception as e:
 255 |             print(f"⚠️ Failed to generate PnL over time plot: {e}")
 256 |             
 257 |         try:
 258 |             # Relative performance plot
 259 |             relative_plot_path = self.output_dir / 'relative_performance.html'
 260 |             generate_relative_performance(self.method_results, relative_plot_path, "Agent's Performance PnL over Baseline")
 261 |             output_files['relative_performance'] = relative_plot_path
 262 |         except Exception as e:
 263 |             print(f"⚠️ Failed to generate relative performance plot: {e}")
 264 | 
 265 |         return output_files
 266 |     def _create_summary_table(self, methods: List[str]) -> List[Dict]:
 267 |         """Create the main summary comparison table."""
 268 |         table_data = []
 269 |         
 270 |         for method in methods:
 271 |             if method not in self.method_results:
 272 |                 continue
 273 |                 
 274 |             result = self.method_results[method]
 275 |             
 276 |             # Handle both old and new data structures
 277 |             if 'academic_metrics' in result:
 278 |                 # New academic approach structure
 279 |                 academic_metrics = result['academic_metrics']
 280 |                 financial = academic_metrics.get('financial', {})
 281 |                 rl_dynamics = academic_metrics.get('rl_dynamics', {})
 282 |                 behavioral = academic_metrics.get('behavioral', {})
 283 |             else:
 284 |                 # Legacy structure
 285 |                 financial = result.get('financial', {})
 286 |                 rl_dynamics = result.get('rl_dynamics', {})
 287 |                 behavioral = result.get('behavioral', {})
 288 |             
 289 |             # Method display names
 290 |             display_names = {
 291 |                 'no_distillation': 'Baseline (No Distillation)',
 292 |                 'offline_pkt': 'PKT Distillation',
 293 |                 'rl_ttm': 'TTM Distillation'
 294 |             }
 295 |             
 296 |             row = {
 297 |                 'Method': display_names.get(method, method),
 298 |                 'Sharpe Ratio': f"{financial.get('sharpe_ratio', {}).get('mean', 0):.3f}",
 299 |                 'Sortino Ratio': f"{financial.get('sortino_ratio', {}).get('mean', 0):.3f}",
 300 |                 'Calmar Ratio': f"{financial.get('calmar_ratio', {}).get('mean', 0):.3f}",
 301 |                 'Max Drawdown (%)': f"{financial.get('max_drawdown', {}).get('mean', 0):.2f}",
 302 |                 'Profit Factor': f"{financial.get('profit_factor', {}).get('mean', 0):.3f}",
 303 |                 'Turnover (Trades/Year)': f"{behavioral.get('trade_frequency', {}).get('mean', 0):.0f}",
 304 |                 'Policy Entropy': f"{rl_dynamics.get('entropy_mean', {}).get('mean', 0):.4f} ± {rl_dynamics.get('entropy_std', {}).get('mean', 0):.4f}",
 305 |                 'Training Stability': f"{rl_dynamics.get('training_stability', {}).get('mean', 0):.3f}"
 306 |             }
 307 |             
 308 |             table_data.append(row)
 309 |         
 310 |         return table_data
 311 |     
 312 |     def _create_financial_analysis(self, methods: List[str]) -> str:
 313 |         """Create financial performance analysis text."""
 314 |         analysis = []
 315 |         analysis.append("## Financial Performance Analysis\n")
 316 |         
 317 |         # Find best performing method
 318 |         best_sharpe = 0
 319 |         best_method = None
 320 |         
 321 |         for method in methods:
 322 |             if method in self.method_results:
 323 |                 sharpe = self.method_results[method]['financial'].sharpe_ratio
 324 |                 if sharpe > best_sharpe:
 325 |                     best_sharpe = sharpe
 326 |                     best_method = method
 327 |         
 328 |         if best_method:
 329 |             display_name = {'no_distillation': 'Baseline', 'offline_pkt': 'PKT', 'rl_ttm': 'TTM'}.get(best_method, best_method)
 330 |             analysis.append(f"The {display_name} method achieved the highest risk-adjusted returns with a Sharpe ratio of {best_sharpe:.3f}.")
 331 |             
 332 |             # Add specific analysis for TTM vs PKT
 333 |             if best_method == 'rl_ttm':
 334 |                 analysis.append("\nThe superior performance of TTM can be attributed to its effective regularization through ensemble teacher matching, ")
 335 |                 analysis.append("which prevented overfitting and led to better generalization on out-of-sample data.")
 336 |             elif best_method == 'offline_pkt':
 337 |                 analysis.append("\nPKT distillation showed strong performance through effective knowledge transfer from the teacher ensemble.")
 338 |         
 339 |         return "\n".join(analysis)
 340 |     
 341 |     def _create_rl_dynamics_analysis(self, methods: List[str]) -> str:
 342 |         """Create RL dynamics analysis focusing on policy entropy and training stability."""
 343 |         analysis = []
 344 |         analysis.append("## Reinforcement Learning Dynamics Analysis\n")
 345 |         
 346 |         # Compare policy entropy - key metric for TTM
 347 |         entropy_comparison = []
 348 |         for method in methods:
 349 |             if method in self.method_results:
 350 |                 rl_metrics = self.method_results[method]['rl_dynamics']
 351 |                 display_name = {'no_distillation': 'Baseline', 'offline_pkt': 'PKT', 'rl_ttm': 'TTM'}.get(method, method)
 352 |                 entropy_comparison.append((display_name, rl_metrics.entropy_mean, rl_metrics.entropy_trend))
 353 |         
 354 |         analysis.append("### Policy Entropy Analysis")
 355 |         analysis.append("Policy entropy is a critical indicator of exploration vs exploitation balance and regularization effectiveness.\n")
 356 |         
 357 |         for name, entropy, trend in entropy_comparison:
 358 |             analysis.append(f"- **{name}**: Mean entropy = {entropy:.4f}, Trend = {trend}")
 359 |         
 360 |         # Highlight TTM's regularization effect
 361 |         if 'rl_ttm' in self.method_results:
 362 |             ttm_entropy = self.method_results['rl_ttm']['rl_dynamics'].entropy_mean
 363 |             analysis.append(f"\nThe TTM method shows {'higher' if ttm_entropy > 0.5 else 'controlled'} policy entropy, ")
 364 |             analysis.append("indicating effective regularization that prevents overconfident policies and maintains exploration capability.")
 365 |         
 366 |         return "\n".join(analysis)
 367 |     
 368 |     def _create_behavioral_analysis(self, methods: List[str]) -> str:
 369 |         """Create behavioral and trading statistics analysis."""
 370 |         analysis = []
 371 |         analysis.append("## Behavioral & Trading Statistics Analysis\n")
 372 |         
 373 |         analysis.append("### Trading Behavior Characteristics")
 374 |         
 375 |         for method in methods:
 376 |             if method in self.method_results:
 377 |                 behavioral = self.method_results[method]['behavioral']
 378 |                 display_name = {'no_distillation': 'Baseline', 'offline_pkt': 'PKT', 'rl_ttm': 'TTM'}.get(method, method)
 379 |                 
 380 |                 analysis.append(f"\n**{display_name}**:")
 381 |                 analysis.append(f"- Trading Frequency: {behavioral.get('trade_frequency', {}).get('mean', 0):.0f} trades/year")
 382 |                 analysis.append(f"- Average Holding Period: {behavioral.get('avg_holding_period', {}).get('mean', 0):.1f} time units")
 383 |                 analysis.append(f"- Position Concentration: {behavioral.get('position_concentration', {}).get('mean', 0):.3f}")
 384 |                 analysis.append(f"- VaR (95%): {behavioral.get('var_95', {}).get('mean', 0):.3f}")
 385 |         
 386 |         return "\n".join(analysis)
 387 |     
 388 |     def _generate_conclusions(self, methods: List[str]) -> str:
 389 |         """Generate final conclusions and architectural recommendations."""
 390 |         conclusions = []
 391 |         conclusions.append("## Conclusions and Architectural Decision\n")
 392 |         
 393 |         # Find the best method based on comprehensive metrics
 394 |         best_method = self._determine_best_method(methods)
 395 |         
 396 |         if best_method:
 397 |             display_name = {'no_distillation': 'Baseline', 'offline_pkt': 'PKT Distillation', 'rl_ttm': 'TTM Distillation'}.get(best_method, best_method)
 398 |             
 399 |             conclusions.append(f"Based on the comprehensive empirical evidence across financial performance, ")
 400 |             conclusions.append(f"RL dynamics, and behavioral analysis, **{display_name}** demonstrates superior performance.")
 401 |             
 402 |             if best_method == 'rl_ttm':
 403 |                 conclusions.append("\n### TTM Superiority Analysis")
 404 |                 conclusions.append("The Transformed Teacher Matching (TTM) method shows clear advantages:")
 405 |                 conclusions.append("1. **Higher risk-adjusted returns** with better Sharpe and Calmar ratios")
 406 |                 conclusions.append("2. **Effective regularization** evidenced by stable policy entropy")
 407 |                 conclusions.append("3. **Robust generalization** with lower maximum drawdown")
 408 |                 conclusions.append("4. **Stable training dynamics** with consistent convergence")
 409 |                 
 410 |                 conclusions.append(f"\n**Architectural Decision**: TTM will be adopted as the standard distillation technique for all future model development.")
 411 |             
 412 |         conclusions.append("\n### Statistical Robustness")
 413 |         conclusions.append("All results are reported as mean ± standard deviation across multiple independent training runs with different random seeds, ")
 414 |         conclusions.append("ensuring statistical robustness and reproducibility of findings.")
 415 |         
 416 |         return "\n".join(conclusions)
 417 |     
 418 |     def _determine_best_method(self, methods: List[str]) -> Optional[str]:
 419 |         """Determine the best method based on comprehensive scoring."""
 420 |         if not methods or not self.method_results:
 421 |             return None
 422 |         
 423 |         scores = {}
 424 |         
 425 |         for method in methods:
 426 |             if method not in self.method_results:
 427 |                 continue
 428 |                 
 429 |             result = self.method_results[method]
 430 |             
 431 |             # Handle both old and new data structures
 432 |             if 'academic_metrics' in result:
 433 |                 # New academic approach structure
 434 |                 academic_metrics = result['academic_metrics']
 435 |                 financial = academic_metrics.get('financial', {})
 436 |                 rl_dynamics = academic_metrics.get('rl_dynamics', {})
 437 |             else:
 438 |                 # Legacy structure
 439 |                 financial = result.get('financial', {})
 440 |                 rl_dynamics = result.get('rl_dynamics', {})
 441 |             
 442 |             # Composite score (weighted)
 443 |             score = (
 444 |                 financial.get('sharpe_ratio', {}).get('mean', 0) * 0.4 +  # Financial performance (40%)
 445 |                 financial.get('calmar_ratio', {}).get('mean', 0) * 0.3 +  # Risk-adjusted return (30%)
 446 |                 rl_dynamics.get('training_stability', {}).get('mean', 0) * 0.2 +  # Training stability (20%)
 447 |                 rl_dynamics.get('entropy_mean', {}).get('mean', 0) * 0.1  # Regularization quality (10%)
 448 |             )
 449 |             
 450 |             scores[method] = score
 451 |         
 452 |         return max(scores.keys(), key=lambda k: scores[k]) if scores else None
 453 |     
 454 |     def _save_summary_csv(self, methods: List[str], output_path: Path) -> None:
 455 |         """Save summary comparison table as CSV."""
 456 |         table_data = self._create_summary_table(methods)
 457 |         df = pd.DataFrame(table_data)
 458 |         df.to_csv(output_path, index=False)
 459 |         print(f"Summary table saved to: {output_path}")
 460 |     
 461 |     def _generate_visualizations(self, methods: List[str]) -> Dict[str, Path]:
 462 |         """Generate key visualizations for the report."""
 463 |         viz_files = {}
 464 |         
 465 |         # 1. Performance comparison bar chart
 466 |         performance_chart = self._create_performance_comparison_chart(methods)
 467 |         if performance_chart:
 468 |             viz_files['performance_comparison'] = performance_chart
 469 |         
 470 |         # 2. Policy entropy comparison
 471 |         entropy_chart = self._create_entropy_comparison_chart(methods)
 472 |         if entropy_chart:
 473 |             viz_files['entropy_comparison'] = entropy_chart
 474 |         
 475 |         # 3. Risk-return scatter plot
 476 |         risk_return_chart = self._create_risk_return_scatter(methods)
 477 |         if risk_return_chart:
 478 |             viz_files['risk_return_scatter'] = risk_return_chart
 479 |         
 480 |         return viz_files
 481 |     
 482 |     def _create_performance_comparison_chart(self, methods: List[str]) -> Optional[Path]:
 483 |         """Create performance comparison bar chart."""
 484 |         try:
 485 |             metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
 486 |             method_names = []
 487 |             sharpe_values = []
 488 |             sortino_values = []
 489 |             calmar_values = []
 490 |             
 491 |             for method in methods:
 492 |                 if method in self.method_results:
 493 |                     financial = self.method_results[method]['financial']
 494 |                     display_name = {'no_distillation': 'Baseline', 'offline_pkt': 'PKT', 'rl_ttm': 'TTM'}.get(method, method)
 495 |                     
 496 |                     method_names.append(display_name)
 497 |                     sharpe_values.append(financial.get('sharpe_ratio', {}).get('mean', 0))
 498 |                     sortino_values.append(financial.get('sortino_ratio', {}).get('mean', 0))
 499 |                     calmar_values.append(financial.get('calmar_ratio', {}).get('mean', 0))
 500 |             
 501 |             if not method_names:
 502 |                 return None
 503 |             
 504 |             # Create the plot
 505 |             fig, ax = plt.subplots(figsize=(12, 8))
 506 |             x = np.arange(len(method_names))
 507 |             width = 0.25
 508 |             
 509 |             bars1 = ax.bar(x - width, sharpe_values, width, label='Sharpe Ratio', alpha=0.8)
 510 |             bars2 = ax.bar(x, sortino_values, width, label='Sortino Ratio', alpha=0.8)
 511 |             bars3 = ax.bar(x + width, calmar_values, width, label='Calmar Ratio', alpha=0.8)
 512 |             
 513 |             ax.set_xlabel('Method', fontsize=12, fontweight='bold')
 514 |             ax.set_ylabel('Ratio Value', fontsize=12, fontweight='bold')
 515 |             ax.set_title('Risk-Adjusted Performance Comparison', fontsize=14, fontweight='bold')
 516 |             ax.set_xticks(x)
 517 |             ax.set_xticklabels(method_names)
 518 |             ax.legend()
 519 |             ax.grid(True, alpha=0.3)
 520 |             
 521 |             # Add value labels on bars
 522 |             for bars in [bars1, bars2, bars3]:
 523 |                 for bar in bars:
 524 |                     height = bar.get_height()
 525 |                     ax.annotate(f'{height:.3f}',
 526 |                                xy=(bar.get_x() + bar.get_width() / 2, height),
 527 |                                xytext=(0, 3),
 528 |                                textcoords="offset points",
 529 |                                ha='center', va='bottom',
 530 |                                fontsize=9)
 531 |             
 532 |             plt.tight_layout()
 533 |             
 534 |             output_path = self.output_dir / 'performance_comparison.png'
 535 |             plt.savefig(output_path, dpi=300, bbox_inches='tight')
 536 |             plt.close()
 537 |             
 538 |             return output_path
 539 |             
 540 |         except Exception as e:
 541 |             print(f"Error creating performance comparison chart: {e}")
 542 |             return None
 543 |     
 544 |     def _create_entropy_comparison_chart(self, methods: List[str]) -> Optional[Path]:
 545 |         """Create policy entropy comparison chart - key for TTM analysis."""
 546 |         try:
 547 |             method_names = []
 548 |             entropy_means = []
 549 |             entropy_stds = []
 550 |             
 551 |             for method in methods:
 552 |                 if method in self.method_results:
 553 |                     rl_metrics = self.method_results[method]['rl_dynamics']
 554 |                     display_name = {'no_distillation': 'Baseline', 'offline_pkt': 'PKT', 'rl_ttm': 'TTM'}.get(method, method)
 555 |                     
 556 |                     method_names.append(display_name)
 557 |                     entropy_means.append(rl_metrics.entropy_mean)
 558 |                     entropy_stds.append(rl_metrics.entropy_std)
 559 |             
 560 |             if not method_names:
 561 |                 return None
 562 |             
 563 |             # Create the plot
 564 |             fig, ax = plt.subplots(figsize=(10, 6))
 565 |             
 566 |             bars = ax.bar(method_names, entropy_means, yerr=entropy_stds,
 567 |                          capsize=5, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(method_names)])
 568 |             
 569 |             ax.set_xlabel('Method', fontsize=12, fontweight='bold')
 570 |             ax.set_ylabel('Policy Entropy', fontsize=12, fontweight='bold')
 571 |             ax.set_title('Policy Entropy Comparison - Regularization Effectiveness', fontsize=14, fontweight='bold')
 572 |             ax.grid(True, alpha=0.3)
 573 |             
 574 |             # Add value labels on bars
 575 |             for bar, mean, std in zip(bars, entropy_means, entropy_stds):
 576 |                 height = bar.get_height()
 577 |                 ax.annotate(f'{mean:.4f}±{std:.4f}',
 578 |                            xy=(bar.get_x() + bar.get_width() / 2, height),
 579 |                            xytext=(0, 3),
 580 |                            textcoords="offset points",
 581 |                            ha='center', va='bottom',
 582 |                            fontsize=10)
 583 |             
 584 |             plt.tight_layout()
 585 |             
 586 |             output_path = self.output_dir / 'policy_entropy_comparison.png'
 587 |             plt.savefig(output_path, dpi=300, bbox_inches='tight')
 588 |             plt.close()
 589 |             
 590 |             return output_path
 591 |             
 592 |         except Exception as e:
 593 |             print(f"Error creating entropy comparison chart: {e}")
 594 |             return None
 595 |     
 596 |     def _create_risk_return_scatter(self, methods: List[str]) -> Optional[Path]:
 597 |         """Create risk-return scatter plot."""
 598 |         try:
 599 |             method_names = []
 600 |             returns = []
 601 |             risks = []
 602 |             
 603 |             for method in methods:
 604 |                 if method in self.method_results:
 605 |                     financial = self.method_results[method]['financial']
 606 |                     display_name = {'no_distillation': 'Baseline', 'offline_pkt': 'PKT', 'rl_ttm': 'TTM'}.get(method, method)
 607 |                     
 608 |                     method_names.append(display_name)
 609 |                     returns.append(financial.get('cumulative_pnl', {}).get('mean', 0))
 610 |                     risks.append(financial.get('max_drawdown', {}).get('mean', 0))
 611 |             
 612 |             if not method_names:
 613 |                 return None
 614 |             
 615 |             # Create the plot
 616 |             fig, ax = plt.subplots(figsize=(10, 8))
 617 |             
 618 |             colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(method_names)]
 619 |             scatter = ax.scatter(risks, returns, c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)
 620 |             
 621 |             # Add method labels
 622 |             for i, name in enumerate(method_names):
 623 |                 ax.annotate(name, (risks[i], returns[i]),
 624 |                            xytext=(10, 10), textcoords='offset points',
 625 |                            fontsize=12, fontweight='bold')
 626 |             
 627 |             ax.set_xlabel('Risk (Max Drawdown %)', fontsize=12, fontweight='bold')
 628 |             ax.set_ylabel('Return (Cumulative PnL)', fontsize=12, fontweight='bold')
 629 |             ax.set_title('Risk-Return Profile Comparison', fontsize=14, fontweight='bold')
 630 |             ax.grid(True, alpha=0.3)
 631 |             
 632 |             # Add quadrant labels
 633 |             ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
 634 |             ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
 635 |             
 636 |             plt.tight_layout()
 637 |             
 638 |             output_path = self.output_dir / 'risk_return_scatter.png'
 639 |             plt.savefig(output_path, dpi=300, bbox_inches='tight')
 640 |             plt.close()
 641 |             
 642 |             return output_path
 643 |             
 644 |         except Exception as e:
 645 |             print(f"Error creating risk-return scatter plot: {e}")
 646 |             return None

```
```