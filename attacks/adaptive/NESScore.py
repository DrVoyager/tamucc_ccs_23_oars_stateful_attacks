from IPython import embed
from abc import abstractmethod
import torch
import math
from tqdm.auto import tqdm
from IPython import embed
from attacks.Attack import Attack
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import logging


class NESScore(Attack):
    def __init__(self, model, model_config, attack_config):
        self.image_number = 0
        # self.highest_probabilities = []
        self.highest_probabilities_all = []
        # self.second_highest_probabilities = []
        self.second_highest_probabilities_all = []
        self.cache_values_all = []
        self.output_folder = "imagewise"
        self.avg_loss_all = []

        self.index_label = []
        self.currentIndex = 0
        self.plotIndex = []
        super().__init__(model, model_config, attack_config)

    def attack_untargeted(self, x, y):

        print("NES untargeted attack started..")
        logging.info("NES untargeted attack started..")

        # original image loss
        probs_orig, is_cache = self.model(x)
        self.update_all_probabilities(probs_orig, is_cache)
        loss_orig = self.loss(probs_orig, y)

        print(f"Computed loss for Original Image, loss = {loss_orig}")
        logging.info(f"Computed loss for Original Image, loss = {loss_orig}")

        # initialize
        x_adv = x.detach()
        x_adv = x_adv + torch.FloatTensor(*x.shape).uniform_(
            -self.attack_config["eps"], self.attack_config["eps"]
        )

        # variables and bookkeeping
        if self.attack_config["adaptive"]["bs_grad_var"]:
            var = self.binary_search_gradient_estimation_variance(x)
        else:
            var = self.attack_config["var"]

        if self.attack_config["adaptive"]["bs_min_ss"]:
            bs_min_step_size = self.binary_search_minimum_step_size(x)
            # bs_min_step_size = self.interval_search_minimum_step_size(x)
        else:
            bs_min_step_size = 0

        step_size = self.attack_config["step_size"]
        prev_grad_est = None
        prev_x_adv = None
        loss_history = []
        step_attempts = 0

        print(
            f"Before iteration process,  Variance : {var}, Step size : {step_size}, Momentum: {self.attack_config['momentum']}"
        )
        logging.info(
            f"Before iteration process, Variance : {var}, Step size : {step_size}, Momentum: {self.attack_config['momentum']}"
        )

        # attack loop
        pbar = tqdm(range(self.attack_config["max_iter"]), colour="red", leave=True)

        print("Begin Iterations - Adapting Step Size..")
        logging.info("Begin Iterations - Adapting Step Size..")

        for itr in pbar:
            # estimate gradient

            self.plotIndex.append(self.currentIndex)
            self.index_label.append(f"Iteration {itr +1 }")

            avg_loss, grad_est = self.estimate_gradient(
                x_adv, y, self.attack_config["num_dirs"], var
            )

            self.avg_loss_all.append(avg_loss)

            # gradient momentum
            if step_attempts == 0:
                if prev_grad_est is not None:
                    grad_est = (
                        self.attack_config["momentum"] * prev_grad_est
                        + (1 - self.attack_config["momentum"]) * grad_est
                    )
            prev_grad_est = grad_est
            prev_x_adv = x_adv

            # anneal step size
            loss_history.append(avg_loss)
            loss_history = loss_history[-self.attack_config["plateau_length"] :]
            if (
                loss_history[-1] > loss_history[0]
                and len(loss_history) == self.attack_config["plateau_length"]
            ):
                if step_size > self.attack_config["min_step_size"]:
                    step_size = max(
                        step_size / self.attack_config["plateau_drop"],
                        self.attack_config["min_step_size"],
                    )
                loss_history = []

            if self.attack_config["adaptive"]["bs_min_ss"]:
                step_size = max(step_size, bs_min_step_size)

            # step
            x_adv = x_adv + step_size * grad_est.sign()

            eta = torch.clamp(
                x_adv - x, min=-self.attack_config["eps"], max=self.attack_config["eps"]
            )
            x_adv = torch.clamp(x + eta, min=0, max=1).detach_()

            step_attempts += 1

            print(
                f"Current Iteration : {itr +1} Step attempts : {step_attempts} Step Size : {step_size:.6f} "
            )
            logging.info(
                f"Current Iteration : {itr +1} Step attempts : {step_attempts} Step Size : {step_size:.6f} "
            )

            curr_probs, is_cache = self.model(x_adv)
            self.update_all_probabilities(curr_probs, is_cache)

            if (
                is_cache[0]
                and step_attempts < self.attack_config["adaptive"]["step_max_attempts"]
            ):
                print("Step movement failure. Retrying.")
                logging.info("Step movement failure. Retrying.")
                x_adv = prev_x_adv
                continue
            elif (
                is_cache[0]
                and step_attempts >= self.attack_config["adaptive"]["step_max_attempts"]
            ):
                self.plotIndex.append(self.currentIndex - 1)
                self.index_label.append("ADV Image Failure")
                # self.writeto_csv()
                self.plot_iterative_probs_each_image()

                logging.info("Step movement failure.")
                print("Step movement failure.")
                self.end("Step movement failure.")

            step_attempts = 0
            curr_label = torch.argmax(curr_probs, dim=1)
            curr_loss = self.loss(curr_probs, y)

            # logging
            # pbar.set_description(
            #     f"Label : {curr_label.item()}/{y.item()}| Loss : {loss_orig:.8f}/{curr_loss:.8f} | Cache Hits : {self.get_cache_hits()}/{self.get_total_queries()} | Step Size : {step_size:.6f}"
            # )
            print(
                f"Current Label / True Label : {curr_label.item()}/{y.item()}|  Current Loss : {curr_loss:.8f} | Cache Hits : {self.get_cache_hits()}/{self.get_total_queries()} | Step Size : {step_size:.6f}"
            )
            logging.info(
                f"Current Label / True Label : {curr_label.item()}/{y.item()}| Current Loss : {curr_loss:.8f} | Cache Hits : {self.get_cache_hits()}/{self.get_total_queries()} | Step Size : {step_size:.6f}"
            )

            if curr_label.item() != y.item():
                self.plotIndex.append(self.currentIndex - 1)
                self.index_label.append("ADV Image Success")
                print("ADVERSARIAL IMAGE GENERATED")
                logging.info("ADEVERSARIAL IMAGE GENERATED")
                break

        self.writeto_csv()
        self.plot_iterative_probs_each_image()

        print(f"Iteration Process Completed, current iteration={itr +1}")
        logging.info(f"Iteration Process Completed , current iteration={itr +1}")
        print("NES untargeted attack completed")
        logging.info("NES untargeted attack completed")

        return x_adv

    def attack_targeted(self, x, y, x_adv):

        print("NES targeted attack started..")
        logging.info("NES targeted attack started..")

        # original image loss
        probs_orig, is_cache = self.model(x)
        self.update_all_probabilities(probs_orig, is_cache)
        loss_orig = self.loss(probs_orig, y)
        print(f"Computed loss for Original Image {loss_orig}")
        logging.info(f"Computed loss for Original Image {loss_orig}")

        # initialize
        x_adv = x.detach()
        x_adv = x_adv + torch.FloatTensor(*x.shape).uniform_(
            -self.attack_config["eps"], self.attack_config["eps"]
        )

        # variables and bookkeeping
        if self.attack_config["adaptive"]["bs_grad_var"]:
            var = self.binary_search_gradient_estimation_variance(x)
        else:
            var = self.attack_config["var"]

        if self.attack_config["adaptive"]["bs_min_ss"]:
            bs_min_step_size = self.binary_search_minimum_step_size(x)
            step_query_interval = bs_min_step_size / self.attack_config["min_step_size"]
            step_query_interval *= self.attack_config["adaptive"]["bs_min_ss_hit_rate"]
            step_query_interval = math.ceil(step_query_interval)
        else:
            step_query_interval = 1

        step_size = self.attack_config["step_size"]
        prev_grad_est = None
        loss_history = []
        step_attempts = 0

        print(f"Variance before Iteration proccess {var}")
        logging.info(f"Variance before Iteration proccess {var}")
        print(f"Step size before Iteration process {step_size}")
        logging.info(f"Step size before Iteration process {step_size}")
        print(f"StepQuery Interval before Iteration proccess {step_query_interval}")
        logging.info(
            f"Step_Query_Interval before Iteration proccess {step_query_interval}"
        )

        # print(f"Momemntum = { self.attack_config["momentum"] }")
        # logging.info(f"Momemntum = { self.attack_config["momentum"] }")

        # attack loop
        pbar = tqdm(range(self.attack_config["max_iter"]), colour="red", leave=True)

        print("Begin Iterative process - Adapting Step Size..")
        logging.info("Begin Iterative process - Adapting Step Size..")

        for _ in pbar:
            # estimate gradient
            self.plotIndex.append(self.currentIndex)
            self.index_label.append(f"Iteration {_ + 1}")

            avg_loss, grad_est = self.estimate_gradient(
                x_adv, y, self.attack_config["num_dirs"], var
            )
            self.avg_loss_all.append(avg_loss)

            # gradient momentum
            if step_attempts == 0:
                if prev_grad_est is not None:
                    grad_est = (
                        self.attack_config["momentum"] * prev_grad_est
                        + (1 - self.attack_config["momentum"]) * grad_est
                    )
            prev_grad_est = grad_est
            prev_x_adv = x_adv

            # anneal step size
            loss_history.append(avg_loss)
            loss_history = loss_history[-self.attack_config["plateau_length"] :]
            if (
                loss_history[-1] > loss_history[0]
                and len(loss_history) == self.attack_config["plateau_length"]
            ):
                if step_size > self.attack_config["min_step_size"]:
                    step_size = max(
                        step_size / self.attack_config["plateau_drop"],
                        self.attack_config["min_step_size"],
                    )
                loss_history = []

            print(
                f"Current Iteration : {_ +1}  Step Size : {step_size:.6f} Step Query Interval : {step_query_interval} "
            )
            logging.info(
                f"Current Iteration : {_ +1}  Step Size : {step_size:.6f} Step Query Interval : {step_query_interval}"
            )

            # STEP SIZE ANNEALING
            x_adv = x_adv - step_size * grad_est.sign()
            eta = torch.clamp(
                x_adv - x, min=-self.attack_config["eps"], max=self.attack_config["eps"]
            )
            x_adv = torch.clamp(x + eta, min=0, max=1).detach_()

            if _ % step_query_interval == 0:
                step_attempts += 1
                curr_probs, is_cache = self.model(x_adv)
                self.update_all_probabilities(curr_probs, is_cache)
                if (
                    is_cache[0]
                    and step_attempts
                    < self.attack_config["adaptive"]["step_max_attempts"]
                ):
                    print("Step movement failure, retrying..")
                    logging.info("Step movement failure, retrying..")
                    x_adv = prev_x_adv
                    continue
                elif (
                    is_cache[0]
                    and step_attempts
                    >= self.attack_config["adaptive"]["step_max_attempts"]
                ):

                    self.plotIndex.append(self.currentIndex - 1)
                    self.index_label.append("ADV Img failure")
                    # self.writeto_csv()
                    self.plot_iterative_probs_each_image()

                    print("Step movement failure, Reached Max attempts.")
                    logging.info("Step movement failure, Reached Max attempts.")
                    self.end("Step movement failure.")

                step_attempts = 0
                curr_label = torch.argmax(curr_probs, dim=1)
                curr_loss = self.loss(curr_probs, y)

                # logging
                # pbar.set_description(f"Label : {y.item()}/{curr_label.item()}| Loss : {loss_orig:.8f}/{curr_loss:.8f} "
                #     f"| Cache Hits : {
                #         self.get_cache_hits()}/{self.get_total_queries()} "
                #     f"| Var : {var:.6f} | Step Size : {
                #         step_size:.6f} | Step Query Interval : {step_query_interval}"
                #     f"| Step Attempts : {step_attempts}"
                # )
                print(
                    f"Targeted Label: {y.item()} Predicted Label: {curr_label.item()}|  Current Loss : {curr_loss:.8f} "
                    f"| Cache Hits : {self.get_cache_hits()}/{self.get_total_queries()} "
                    f"| Var : {var:.6f} | Step Size : {step_size:.6f} | Step Query Interval : {step_query_interval}"
                    f"| Step Attempts : {step_attempts}"
                )
                logging.info(
                    f"Targeted Label : {y.item()} Predicted Label : {curr_label.item()}|  Current Loss : {curr_loss:.8f} "
                    f"| Cache Hits : {self.get_cache_hits()}/{self.get_total_queries()} "
                    f"| Var : {var:.6f} | Step Size : {step_size:.6f} | Step Query Interval : {step_query_interval}"
                    f"| Step Attempts : {step_attempts}"
                )
                if curr_label.item() == y.item():
                    self.plotIndex.append(self.currentIndex - 1)
                    self.index_label.append("ADV Image Generated")

                    print("ADV IMAGE GENERATED")
                    logging.info("ADV IMAGE GENERATED")
                    break

        self.writeto_csv()
        self.plot_iterative_probs_each_image()
        print(f"Iteration Process Completed, current iteration={_ +1}")
        logging.info(f"Iteration Process Completed , current iteration={_ +1}")

        print("NES targeted attack completed")
        logging.info("NES targeted attack completed")

        return x_adv

    def writeto_csv(self):
        # filename=f'testfile{self.image_number}.csv'
        directory = "csvfiles"
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"testfile{self.image_number}.csv")
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "CurrentIndex",
                    "IndexLabel",
                    "NES_Logits_True/False",
                    "SortedList_First_Element",
                    "SortedList_Second_Element",
                    "PredictedLabel_first",
                    "Predicted_Label_second",
                    "Margin Loss",
                ]
            )
            i = 1
            z = 0
            for value1, value2, value3 in zip(
                self.highest_probabilities_all,
                self.second_highest_probabilities_all,
                self.cache_values_all,
            ):

                if i - 1 in self.plotIndex:
                    writer.writerow(
                        [
                            str(i - 1),
                            self.index_label[z],
                            str(value3),
                            value1[1],
                            value2[1],
                            value1[0],
                            value2[0],
                            value1[1] - value2[1],
                        ]
                    )
                    z = z + 1
                else:
                    writer.writerow(
                        [
                            " ",
                            " ",
                            str(value3),
                            value1[1],
                            value2[1],
                            value1[0],
                            value2[0],
                            value1[1] - value2[1],
                        ]
                    )
                i = i + 1

    def loss(self, probs, y):
        loss = torch.nn.functional.nll_loss(torch.log(probs), y)
        return loss

    def binary_search_minimum_step_size(self, x):

        print(
            f"Begin binary search for initial step size.. currentIndex={self.currentIndex}"
        )
        logging.info(
            f"Begin binary search for initial step size.. currentIndex={self.currentIndex}"
        )

        self.plotIndex.append(self.currentIndex)
        self.index_label.append("Est_Initial_Step_Size")

        lower = self.attack_config["adaptive"]["bs_min_ss_lower"]
        upper = self.attack_config["adaptive"]["bs_min_ss_upper"]
        ss = upper
        for _ in range(self.attack_config["adaptive"]["bs_min_ss_steps"]):
            mid = (lower + upper) / 2
            cache_hits = 0
            for _ in range(self.attack_config["adaptive"]["bs_min_ss_sample_size"]):
                step = torch.where(
                    torch.rand(*x.shape).to(x.device) < 0.5, -1, 1
                )  # 2 * torch.rand(*x.shape).to(x.device) - 1 #
                noisy_img = x + mid * step
                noisy_img = torch.clamp(noisy_img, min=0, max=1)
                probs, is_cache = self.model(noisy_img)
                self.update_all_probabilities(probs, is_cache)

                if is_cache[0]:
                    cache_hits += 1
            if (
                cache_hits / self.attack_config["adaptive"]["bs_min_ss_sample_size"]
                <= self.attack_config["adaptive"]["bs_min_ss_hit_rate"]
            ):
                ss = mid
                upper = mid
            else:
                lower = mid
                # upper = upper * 1.5
            print(
                f"Step Size : {ss:.6f} | Cache Hits : {cache_hits}/{self.attack_config['adaptive']['bs_min_ss_sample_size']}, upper : {upper:.6f}, lower : {lower:.6f}"
            )
            logging.info(
                f"Step Size : {ss:.6f} | Cache Hits : {cache_hits}/{self.attack_config['adaptive']['bs_min_ss_sample_size']}, upper : {upper:.6f}, lower : {lower:.6f}"
            )

        print(
            f"End Binary Search to estimate initial Step Size.. currentIndex={self.currentIndex}"
        )
        logging.info(
            f"End Binary Search to estimate initial Step Size.. currentIndex={self.currentIndex}"
        )

        return ss

    def smooth_list(self, lst):
        smoothed = []
        for i in range(len(lst)):
            if i == 0:  # First element
                smoothed.append((lst[i] + lst[i + 1]) / 2)
            elif i == len(lst) - 1:  # Last element
                smoothed.append((lst[i - 1] + lst[i]) / 2)
            else:
                smoothed.append((lst[i - 1] + lst[i] + lst[i + 1]) / 3)
        return smoothed

    def binary_search_gradient_estimation_variance(self, x):
        print(
            f"Begin binary search estimating variance.. currentIndex={self.currentIndex}"
        )
        logging.info(
            f"Begin binary search estimating variance.. currentIndex={self.currentIndex}"
        )

        self.plotIndex.append(self.currentIndex)
        self.index_label.append("Estimate_Initial_Variance")

        lower = self.attack_config["adaptive"]["bs_grad_var_lower"]
        upper = self.attack_config["adaptive"]["bs_grad_var_upper"]
        var = upper

        for _ in range(self.attack_config["adaptive"]["bs_grad_var_steps"]):
            mid = (lower + upper) / 2
            cache_hits = 0
            for _ in range(self.attack_config["adaptive"]["bs_grad_var_sample_size"]):
                noise = torch.randn_like(x).to(x.device)
                noise = noise * mid
                noisy_img = x + noise
                noisy_img = torch.clamp(noisy_img, min=0, max=1)
                probs, is_cache = self.model(noisy_img)
                self.update_all_probabilities(probs, is_cache)

                if is_cache[0]:
                    cache_hits += 1
            if (
                cache_hits / self.attack_config["adaptive"]["bs_grad_var_sample_size"]
                <= self.attack_config["adaptive"]["bs_grad_var_hit_rate"]
            ):
                var = mid
                upper = mid
            else:
                lower = mid
            print(
                f"Var : {var:.6f} | Cache Hits : {cache_hits}/{self.attack_config['adaptive']['bs_grad_var_sample_size']}"
            )
            logging.info(
                f"Var : {var:.6f} | Cache Hits : {cache_hits}/{self.attack_config['adaptive']['bs_grad_var_sample_size']}"
            )

        print(f"End Binary Search estimating Variance currentIndex={self.currentIndex}")
        logging.info(
            f"End Binary Search estimating Variance currentIndex={self.currentIndex}"
        )

        return var

    def estimate_gradient(self, x, y, num_dirs, var):

        print(f"Begin gradient estimate.. currentIndex={self.currentIndex}")
        logging.info(f"Begin gradient estimate.. currentIndex={self.currentIndex}")

        grad_est = torch.zeros_like(x)
        losses = []
        num_dirs_goal = num_dirs
        for _ in range(self.attack_config["adaptive"]["grad_max_attempts"]):
            for _ in range(int(num_dirs / 2)):
                dir = torch.randn_like(x) * var
                x_pert = x + dir
                probs, is_cache = self.model(x_pert)
                self.update_all_probabilities(probs, is_cache)

                if is_cache[0]:
                    continue
                neg_dir = -dir
                x_pert = x + neg_dir
                neg_probs, neg_is_cache = self.model(x_pert)
                self.update_all_probabilities(neg_probs, neg_is_cache)

                if neg_is_cache[0]:
                    continue
                loss = self.loss(probs, y)
                neg_loss = self.loss(neg_probs, y)
                losses.append(loss)
                losses.append(neg_loss)
                grad_est += loss * dir + neg_loss * neg_dir

            num_dirs = num_dirs_goal - len(losses)
        if (
            len(losses) != num_dirs_goal
            and not self.attack_config["adaptive"]["grad_est_accept_partial"]
        ):
            logging.info(
                f"Gradient Estimate Failure.. currentIndex={self.currentIndex}"
            )
            self.plotIndex.append(self.currentIndex)
            self.index_label.append(f"Gradient_Est_Failure")
            # self.writeto_csv()
            self.plot_iterative_probs_each_image()
            self.end("Failure in gradient estimation, not enough directions.")
        if len(losses) == 0:
            logging.info(
                f"Gradient Estimate Failure.. currentIndex={self.currentIndex}"
            )
            self.plotIndex.append(self.currentIndex)
            self.index_label.append(f"Gradient_Est_Failure")
            # self.writeto_csv()
            self.plot_iterative_probs_each_image()
            self.end("Failure in gradient estimation, literally zero directions.")
        grad_est /= len(losses)

        print("End Gradient estimate method.. currentIndex=", self.currentIndex)
        logging.info(f"End Gradient estimate method.. currentIndex={self.currentIndex}")

        return torch.stack(losses).mean(), grad_est

    # def update_probabilities(self, curr_probs):
    #     adv_prob = curr_probs.cpu().numpy().tolist()
    #     adv_prob = adv_prob[0]
    #     max_prob_index = np.argmax(adv_prob)
    #     max_prob = adv_prob[max_prob_index]
    #     self.highest_probabilities.append((max_prob_index, max_prob))

    #     temp_prob = adv_prob.copy()
    #     temp_prob[max_prob_index] = -np.inf

    #     second_max_index = np.argmax(temp_prob)
    #     second_max_value = temp_prob[second_max_index]
    #     self.second_highest_probabilities.append((second_max_index, second_max_value))

    def update_all_probabilities(self, curr_probs, is_cache):

        # print("NES logits")
        # print("NES logits, is_cache=", is_cache[0])

        adv_prob = curr_probs.cpu().numpy().tolist()
        adv_prob = adv_prob[0]

        sortedList = adv_prob

        max_prob_index = np.argmax(adv_prob)
        max_prob = adv_prob[max_prob_index]
        self.highest_probabilities_all.append((max_prob_index, max_prob))

        temp_prob = adv_prob.copy()
        temp_prob[max_prob_index] = -np.inf

        second_max_index = np.argmax(temp_prob)
        second_max_value = temp_prob[second_max_index]
        self.second_highest_probabilities_all.append(
            (second_max_index, second_max_value)
        )

        sortedList.sort(reverse=True)
        # print(f"probabilities=[{max_prob},{second_max_value}]")
        # print("sortedList=", sortedList)
        self.cache_values_all.append(is_cache[0])
        self.currentIndex += 1

    def plot_iterative_probs_each_image(self):

        first_ele = [pair[1] for pair in self.highest_probabilities_all]
        seocnd_ele = [pair[1] for pair in self.second_highest_probabilities_all]

        plt.figure(figsize=(30, 6))
        plt.plot(
            range(1, len(self.highest_probabilities_all) + 1),
            first_ele,
            marker="o",
            markersize=4,
            linestyle="-",
            color="b",
            linewidth=0.5,
            label="First Logit",
        )
        plt.plot(
            range(1, len(self.highest_probabilities_all) + 1),
            seocnd_ele,
            marker="x",
            markersize=4,
            linestyle="-",
            color="r",
            linewidth=0.5,
            label="Second Logit",
        )
        # Add labels and title

        # Draw vertical lines for each value in the array
        for idx, highlight_x in enumerate(self.plotIndex):
            plt.axvline(x=highlight_x + 1, color="g", linestyle="--", linewidth=1)
            if idx < 25 or idx == len(self.plotIndex) - 1:
                plt.text(
                    highlight_x + 1,
                    1,
                    self.index_label[idx],
                    color="green",
                    ha="center",
                    va="bottom",
                )

        plt.xlabel("Queries")
        plt.ylabel("Probabilities")
        plt.title("2-D Plot of NES Attack probings")
        adjusted_plotIndex = np.array([i + 1 for i in self.plotIndex])
        xticks = list(plt.xticks()[0])
        max_x_tick = max(xticks)
        min_x_tick = max(min(xticks), 0)

        tick_range = np.arange(min_x_tick, max_x_tick + 25, 25)
        xticks_combined = np.unique(np.concatenate([tick_range, adjusted_plotIndex]))

        xticks_combined = xticks_combined[xticks_combined != 0]
        if 1 not in xticks_combined:
            xticks_combined = np.append(xticks_combined, 1)
        xticks_combined = np.sort(xticks_combined)
        plt.xticks(xticks_combined, fontsize=6)

        plt.legend()
        # Show the plot
        plt.grid(True)

        directory = "imagewise"
        os.makedirs(directory, exist_ok=True)
        save_plt_path = os.path.join(directory, f"graph_{self.image_number}.png")
        self.image_number += 1
        self.highest_probabilities = []
        self.second_highest_probabilities = []
        self.highest_probabilities_all = []
        self.second_highest_probabilities_all = []
        self.cache_values_all = []

        self.currentIndex = 0
        self.index_label = []
        self.plotIndex = []
        plt.savefig(save_plt_path)
        plt.clf


