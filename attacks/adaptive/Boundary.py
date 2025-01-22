import logging
from typing import Optional, Tuple, TYPE_CHECKING
import torch
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format, get_labels_np_array

from models.art_statefuldefense import ArtStatefulDefense
from attacks.Attack import Attack

import os
import uuid

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


def plot_probabilities_vs_iterations(Iteration, trial, top1_probs, top2_probs, top1_labels, top2_labels, title, file_name, save_dir=None):
    plt.figure(figsize=(45, 6))
    
    # Plotting Top 1 probabilities with dots
    plt.plot(trials, top1_probs, linestyle='-', label='Top 1 Probability', marker='o', markersize=3, linewidth=0.5)
    # Plotting Top 2 probabilities with X marks
    plt.plot(trials, top2_probs, linestyle='-', label='Top 2 Probability', marker='x', markersize=3, linewidth=0.5)

    for it, p1, p2, lbl1, lbl2 in zip(trials, top1_probs, top2_probs, top1_labels, top2_labels):
        p1 = float(p1) 
        p2 = float(p2) 

    # Setting custom x-ticks
    x_ticks = np.arange(1, 1000, 100)
    plt.xticks(x_ticks)

    # Set y-ticks with 5 segments
    plt.yticks(np.linspace(0, 1, 5)) 
    plt.ylim(-0.1, 1.1) 

    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth=0.5)

    # Determine the save path
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)
    else:
        file_path = file_name

    # Save the plot
    plt.savefig(file_path)
    
    # Show the plot
    plt.show()

    print(f'Plot saved to: {file_path}')


def save_probabilities_to_csv(trail, top1_probs, top2_probs, top1_labels, top2_labels, cache_hits, margin_loss, noise, file_name, save_dir=None):

    '''
    # Print the contents for debugging
    print(f"Iteration: {Iteration}, Trail: {Trail}, Top 1 Probability: {top1_probs}, Top 2 Probability: {top2_probs}, "
          f"Top 1 Label: {top1_labels}, Top 2 Label: {top2_labels}, Margin Loss: {margin_loss}, "
          f"Cache Preds: {cache_preds}, Attack/Benign: {attack_benign}, Noise: {noise}")
    '''

    # Creating a dictionary for the DataFrame to handle different lengths gracefully
    data = {
        'Trail': trail,
        'Top 1 Probability': top1_probs,
        'Top 2 Probability': top2_probs,
        'Top 1 Label': top1_labels,
        'Top 2 Label': top2_labels,
        'Cache Hits': cache_hits,
        'Margin Loss': margin_loss,
        'Noise': noise
    }
    
    # Creating a DataFrame using the dictionary, where it will automatically align shorter columns
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

    # Determine the save path
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)
    else:
        file_path = file_name

    # Save to CSV
    df.to_csv(file_path, index=False)
    
    print(f'CSV file saved to: {file_path}')



class Boundary(Attack):
    def __init__(self, model, model_config, attack_config):
        self.model_config = model_config
        self.attack_config = attack_config
        self.model_art = ArtStatefulDefense(model=model, device_type='cpu',
                                            input_shape=model_config['state']['input_shape'], loss=None,
                                            nb_classes=attack_config['nb_classes'])
        self.art_attack = BoundaryAttack(estimator=self.model_art, model_config=model_config, batch_size=1,
                                         targeted=False, min_epsilon=attack_config["eps"], attack_config=attack_config)
        self._model = self.model_art._model._model

    def attack_targeted(self, x, y, x_adv, target_class: int = None):
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        assert x_np.shape[0] == 1
        one_hot_labels = torch.zeros((1, self.attack_config['nb_classes']))
        one_hot_labels[0, y_np] = 1

        x_adv_np = self.art_attack.generate(x=x_np, y=one_hot_labels, target_class=1, x_adv_init=x_adv)
        if isinstance(x_adv_np, str):
            self.end(x_adv_np)
        if np.linalg.norm(x_adv_np - x_np) / (x_np.shape[-1] * x_np.shape[-2] * x_np.shape[-3]) ** 0.5 < \
                self.attack_config["eps"]:
            return torch.tensor(x_adv_np)
        return torch.tensor(x_np)


    def attack_untargeted(self, x, y, target_class: int = None):
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()[0]

        assert x_np.shape[0] == 1
        one_hot_labels = torch.zeros((1, self.attack_config['nb_classes']))
        one_hot_labels[0, y_np] = 1

        x_adv_np = self.art_attack.generate(x=x_np, y=one_hot_labels, target_class=1, x_adv_init=None)
        if isinstance(x_adv_np, str):
            self.end(x_adv_np)
        if np.linalg.norm(x_adv_np - x_np) / (x_np.shape[-1] * x_np.shape[-2] * x_np.shape[-3]) ** 0.5 < \
                self.attack_config["eps"]:
            return torch.tensor(x_adv_np)
        return torch.tensor(x_np)

class BoundaryAttack(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "delta",
        "epsilon",
        "step_adapt",
        "max_iter",
        "num_trial",
        "sample_size",
        "init_size",
        "batch_size",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
            self,
            estimator: "CLASSIFIER_TYPE",
            batch_size: int = 64,
            targeted: bool = True,
            delta: float = 0.01,
            epsilon: float = 0.01,
            step_adapt: float = 0.667,
            max_iter: int = 100,
            num_trial: int = 10,
            sample_size: int = 1, 
            init_size: int = 100, 
            min_epsilon: float = 0.0,
            model_config: dict = None,
            attack_config: dict = None,
            verbose: bool = True
    ) -> None:
        super().__init__(estimator=estimator)

        self._targeted = targeted
        self.delta = delta
        self.epsilon = epsilon
        self.step_adapt = step_adapt
        self.max_iter = max_iter
        self.num_trial = num_trial
        self.sample_size = sample_size
        self.init_size = init_size
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_config = model_config
        self.attack_config = attack_config


        self._check_params()

        self.curr_adv: Optional[np.ndarray] = None

        

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, target_class: int = None, **kwargs) -> np.ndarray:


        assert x.shape[0] == 1

        if y is None:
            raise NotImplementedError
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))  # type: ignore

        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=False)

        if y is not None and self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError("This attack has not yet been tested for binary classification with a single output classifier.")

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size)[0], axis=1)
        init_preds = [None] * len(x)
        x_adv_init = [None] * len(x)

        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        x_adv = x.astype(ART_NUMPY_DTYPE)

        # Generate the adversarial samples
        for ind, val in enumerate(tqdm(x_adv, desc="Boundary attack", disable=not self.verbose)):
            if self.targeted:
                out = self._perturb(
                x=val,
                y=y[ind],
                y_p=preds[ind],
                init_pred=init_preds[ind],
                adv_init=x_adv_init[ind],
                clip_min=clip_min,
                clip_max=clip_max,
                )
                if isinstance(out, str):
                    return out
                x_adv[ind] = out
            else:
                out = self._perturb(
                    x=val,
                    y=-1,
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
                if isinstance(out, str):
                    return out
                x_adv[ind] = out

        y = to_categorical(y, self.estimator.nb_classes)

        return x_adv


    def _perturb(
            self,
            x: np.ndarray,
            y: int,
            y_p: int,
            init_pred: int,
            adv_init: np.ndarray,
            clip_min: float,
            clip_max: float,
    ) -> np.ndarray:
        initial_sample = self._init_sample(x, y, y_p, init_pred, adv_init, clip_min, clip_max)
        if isinstance(initial_sample, str):
            return initial_sample

        if initial_sample is None:
            return x

        x_adv = self._attack(
            initial_sample[0],
            x,
            y_p,
            initial_sample[1],
            self.delta,
            self.epsilon,
            clip_min,
            clip_max,
        )

        return x_adv
    
    def _attack(
            self,
            initial_sample: np.ndarray,
            original_sample: np.ndarray,
            y_p: int,
            target: int,
            initial_delta: float,
            initial_epsilon: float,
            clip_min: float,
            clip_max: float,
    ) -> np.ndarray:

        
        x_adv = initial_sample
        self.curr_delta = initial_delta
        self.curr_epsilon = initial_epsilon

        self.curr_adv = x_adv

        best_l2 = 10000000
        best_l2_set = 0
        
        pbar = trange(self.max_iter, leave=False)
        
        # Lists to store results
        potential_top_1_probs = []
        potential_top_1_labels = []
        potential_top_2_probs = []
        potential_top_2_labels = []
        cache_hits = []
        margin_loss = []
        trail=[]
        
        for this_iter in pbar:
            
            for _ in range(self.num_trial):
                
                potential_advs = []
                
                
                for _ in range(self.sample_size):
                    potential_adv = x_adv + self._orthogonal_perturb(self.curr_delta, x_adv, original_sample)
                    potential_adv = np.clip(potential_adv, clip_min, clip_max)
                    potential_advs.append(potential_adv)

                out_preds, cache_preds = self.estimator.predict(np.array(potential_advs), batch_size=self.batch_size)
         
                preds = np.argmax(
                    out_preds,
                    axis=1,
                )

                
                if not self.attack_config["adaptive"]["delta_adjust_rejected_predictions"] and np.sum(np.where(cache_preds[:, 0], 1, 0)) > 0:
                    return "rejection in delta step"

                if self.attack_config["targeted"]:
                    preds = np.where(cache_preds[:, 0], target, preds)
                else:
                    preds = np.where(cache_preds[:, 0], -1, preds)

                if self.attack_config["targeted"]:
                    is_adv = preds == target
                else:
                    is_adv = preds != y_p

                if self.attack_config["adaptive"]["delta_extraction"]:
                    delta_ratio = np.mean(is_adv)
                else:
                    delta_ratio = np.mean(is_adv[~np.array(cache_preds[:, 0])])

                if delta_ratio < 0.2:
                    self.curr_delta *= self.step_adapt
                elif delta_ratio > 0.5:
                    self.curr_delta /= self.step_adapt

                delta_good = is_adv * (preds >= 0) * (~np.array(cache_preds[:, 0]))

                if self.attack_config["adaptive"]["delta_extraction"]:
                    cap = 1
                else:
                    cap = 1

            
                if np.sum(delta_good) >= cap:
                    x_advs = np.array(potential_advs)[
                        np.where(delta_good)[0]]
                    x_advs_delta = x_advs.copy()
                    break
                    
                elif np.sum(cache_preds[:,0])/len(cache_preds[:,0]) > 0.5:
                    if self.attack_config["adaptive"]["delta_extraction"]:
                        self.curr_delta /= self.step_adapt
                

            else:
                return x_adv


            
            
            if self.curr_epsilon > 1:
                self.curr_epsilon = initial_epsilon

            for this_trail in range(self.num_trial):
                
                trail.append(this_trail+1)
                
                perturb = np.repeat(np.array([original_sample]), len(x_advs), axis=0) - x_advs
                perturb *= self.curr_epsilon
                new_potential_advs = x_advs + perturb
                new_potential_advs = np.clip(new_potential_advs, clip_min, clip_max)  
                potential_advs = new_potential_advs

                #print('length of advs:',len(potential_advs))

                output_preds, cache_preds = self.estimator.predict(potential_advs, batch_size=self.batch_size)

                cache_hits.append(cache_preds[0][0])

                sorted_indices = np.argsort(out_preds, axis=1)[:, ::-1]  # Sort indices in descending order

                # Extract top 1 and top 2 probabilities and labels for potential adversarial samples
                top_1_probs = np.max(out_preds, axis=1)[0]
                top_1_labels = np.argmax(out_preds, axis=1)[0]

                top_2_labels = sorted_indices[:, 1][0]  # Second highest labels
                top_2_probs = np.take_along_axis(out_preds, sorted_indices[:, 1:2], axis=1)[0]

                diff = top_1_probs - top_2_probs
                margin_loss.append(diff[0])

                # Append to lists
                potential_top_1_probs.append(top_1_probs)
                potential_top_1_labels.append(top_1_labels)
                
                if len(out_preds) == 1:
                    potential_top_2_probs.append(top_2_probs[0])
                else:
                    potential_top_2_probs.extend(top_2_probs)
                potential_top_2_labels.append(top_2_labels)
                
                
                preds = np.argmax(
                    output_preds,
                    axis=1,
                )

                if not self.attack_config["adaptive"]["eps_adjust_rejected_predictions"] and np.sum(np.where(cache_preds[:, 0], 1, 0)) > 0:
                    return "rejection in eps step"

                if self.attack_config["targeted"]:
                    preds = np.where(cache_preds[:, 0], target, preds)
                else:
                    preds = np.where(cache_preds[:, 0], -1, preds)
                if self.attack_config["targeted"]:
                    is_adv = preds == target
                else:
                    is_adv = preds != y_p

                if self.attack_config["adaptive"]["eps_extraction"]:
                    epsilon_ratio = np.mean(is_adv)
                else:
                    if cache_preds.shape[0] - np.sum(cache_preds[:, 0]) == 0:
                        if np.mean(delta_good) > 0:
                            x_adv = self._best_adv(original_sample, x_advs_delta)
                            self.curr_adv = x_adv
                            break
                        epsilon_ratio = 1
                    else:
                        epsilon_ratio = np.sum(is_adv[~np.array(cache_preds[:, 0])]) / (
                            cache_preds.shape[0] - np.sum(cache_preds[:, 0]))

                if cache_preds.shape[0] == np.sum(cache_preds[:, 0]):
                    if np.mean(delta_good) > 0:
                        x_adv = self._best_adv(original_sample, x_advs_delta)
                        self.curr_adv = x_adv
                        break

                delta_good = is_adv * (preds >= 0)

                if epsilon_ratio < 0.2:
                    self.curr_epsilon *= self.step_adapt
                elif epsilon_ratio > 0.5:
                    self.curr_epsilon /= self.step_adapt

                
                if np.mean(delta_good) > 0:
                    x_adv = self._best_adv(original_sample, potential_advs[np.where(delta_good)[0]])
                    self.curr_adv = x_adv
                    break
            

            l2_normalized = np.linalg.norm(x_adv - original_sample) / (original_sample.shape[-1] * original_sample.shape[-2] * original_sample.shape[-3]) ** 0.5


            if l2_normalized < best_l2: 
                best_l2 = l2_normalized
                best_l2_set = this_iter

            pbar.set_description("Step : {} | L2 Normalized: {} | curr_epsilon: {}".format(this_iter, l2_normalized, self.curr_epsilon))

        
            if l2_normalized < self.attack_config["eps"]:
                print("Breaked out 4")
            
                file_trail_excel_name = f'Trials_{uuid.uuid4().hex}.csv'

                save_probabilities_to_csv(
                    trail,
                    potential_top_1_probs,
                    potential_top_2_probs,
                    potential_top_1_labels,
                    potential_top_2_labels,
                    cache_hits,
                    margin_loss,
                    file_name = file_trail_excel_name,
                    save_dir='/Users/likhithareddykesara/Desktop/ccs_23_oars_stateful_attacks/Results/Blacklight/now',
                )
            
                return x_adv
            
            
            elif self.curr_epsilon < 10e-6:
                return original_sample
                    # Random file name for CSV
        

        file_trail_excel_name = f'Trials_{uuid.uuid4().hex}.csv'

        save_probabilities_to_csv(
            trail,
            potential_top_1_probs,
            potential_top_2_probs,
            potential_top_1_labels,
            potential_top_2_labels,
            cache_hits,
            margin_loss,
            file_name = file_trail_excel_name,
            save_dir='/Users/likhithareddykesara/Desktop/ccs_23_oars_stateful_attacks/Results/Blacklight/now',
        )

        return x_adv

    def _orthogonal_perturb(self, delta: float, current_sample: np.ndarray, original_sample: np.ndarray) -> np.ndarray:
        perturb = np.random.randn(*self.estimator.input_shape).astype(ART_NUMPY_DTYPE)

        perturb /= np.linalg.norm(perturb)
        perturb *= delta * np.linalg.norm(original_sample - current_sample)

        direction = original_sample - current_sample

        direction_flat = direction.flatten()
        perturb_flat = perturb.flatten()

        direction_flat /= np.linalg.norm(direction_flat)
        perturb_flat -= np.dot(perturb_flat, direction_flat.T) * direction_flat
        perturb = perturb_flat.reshape(self.estimator.input_shape)

        hypotenuse = np.sqrt(1 + delta ** 2)
        perturb = ((1 - hypotenuse) * (current_sample - original_sample) + perturb) / hypotenuse
        return perturb

    def _init_sample(
            self,
            x: np.ndarray,
            y: int,
            y_p: int,
            init_pred: int,
            adv_init: np.ndarray,
            clip_min: float,
            clip_max: float,
    ) -> Optional[Tuple[np.ndarray, int]]:
        nprd = np.random.RandomState()
        initial_sample = None

        if self.targeted:
            raise NotImplementedError
            if y == y_p:
                return None
            if adv_init is not None and init_pred == y:
                return adv_init.astype(ART_NUMPY_DTYPE), init_pred
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                output_preds, cache_preds = self.estimator.predict(np.array([random_img]), batch_size=self.batch_size)
                random_class = np.argmax(
                    output_preds,
                    axis=1,
                )[0]
                if not self.attack_config["adaptive"]["init_bypass_rejects"]:
                    return "rejected in initialization"
                if random_class == y and not cache_preds[0][0]:
                    initial_sample = random_img, random_class
                    logger.info("Found initial adversarial image for targeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")
        else:
            if adv_init is not None and init_pred != y_p:
                return adv_init.astype(ART_NUMPY_DTYPE), init_pred
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                class_preds, cache_preds = self.estimator.predict(np.array([random_img]), batch_size=self.batch_size)
                random_class = np.argmax(
                    class_preds,
                    axis=1,
                )[0]
                if not self.attack_config["adaptive"]["init_bypass_rejects"]:
                    return "rejected in initialization"
                if random_class != y_p and not cache_preds[0][0]:
                    initial_sample = random_img, random_class
                    logger.info("Found initial adversarial image for untargeted attack.")
                    break
            else:  
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")
                
        return initial_sample
    
    @staticmethod
    def _best_adv(original_sample: np.ndarray, potential_advs: np.ndarray) -> np.ndarray:
        shape = potential_advs.shape
        min_idx = np.linalg.norm(original_sample.flatten() - potential_advs.reshape(shape[0], -1), axis=1).argmin()
        return potential_advs[min_idx]

    def _check_params(self) -> None:
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")
        if not isinstance(self.num_trial, int) or self.num_trial < 0:
            raise ValueError("The number of trials must be a non-negative integer.")
        if not isinstance(self.sample_size, int) or self.sample_size <= 0:
            raise ValueError("The number of samples must be a positive integer.")
        if not isinstance(self.init_size, int) or self.init_size <= 0:
            raise ValueError("The number of initial trials must be a positive integer.")
        if self.epsilon <= 0:
            raise ValueError("The initial step size for the step towards the target must be positive.")
        if self.delta <= 0:
            raise ValueError("The initial step size for the orthogonal step must be positive.")
        if self.step_adapt <= 0 or self.step_adapt >= 1:
            raise ValueError("The adaptation factor must be in the range (0, 1).")
        if not isinstance(self.min_epsilon, (float, int)) or self.min_epsilon < 0:
            raise ValueError("The minimum epsilon must be non-negative.")
        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
