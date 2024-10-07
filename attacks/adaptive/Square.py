from IPython import embed
from abc import abstractmethod
import torch
from tqdm.auto import tqdm
from IPython import embed
from attacks.Attack import Attack
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import logging


class Square(Attack):
    def __init__(self, model, model_config, attack_config):
        self.image_number = 0
        self.highest_probabilities_all = []
        self.second_highest_probabilities_all = []
        self.cache_values_all = []
        self.output_folder = "imagewise"

        # index labels to understand the intermediate step
        self.index_label = []
        self.currentIndex = 0
        # indexes to understand at which query index we are for that intermediate step
        self.plotIndex = []

        super().__init__(model, model_config, attack_config)

    # def attack_untargeted(self, x, y):
    #     dim = torch.prod(torch.tensor(x.shape[1:]))
    #
    #     def p_selection(step):
    #         step = int(step / self.attack_config["max_iter"] * 10000)
    #         if 10 < step <= 50:
    #             p = self.attack_config["p_init"] / 2
    #         elif 50 < step <= 200:
    #             p = self.attack_config["p_init"] / 4
    #         elif 200 < step <= 500:
    #             p = self.attack_config["p_init"] / 8
    #         elif 500 < step <= 1000:
    #             p = self.attack_config["p_init"] / 16
    #         elif 1000 < step <= 2000:
    #             p = self.attack_config["p_init"] / 32
    #         elif 2000 < step <= 4000:
    #             p = self.attack_config["p_init"] / 64
    #         elif 4000 < step <= 6000:
    #             p = self.attack_config["p_init"] / 128
    #         elif 6000 < step <= 8000:
    #             p = self.attack_config["p_init"] / 256
    #         elif 8000 < step <= 10000:
    #             p = self.attack_config["p_init"] / 512
    #         else:
    #             p = self.attack_config["p_init"]
    #         return p
    #
    #     def margin_loss(x, y):
    #         logits, is_cache = self.model(x)
    #         probs = torch.softmax(logits, dim=1)
    #         top_2_probs, top_2_classes = torch.topk(probs, 2)
    #         if top_2_classes[:, 0] != y:
    #             return 0, is_cache
    #         else:
    #             return top_2_probs[:, 0] - top_2_probs[:, 1], is_cache
    #
    #     # Initialize adversarial example
    #     pert = torch.tensor(np.random.choice([-self.attack_config["eps"], self.attack_config["eps"]],
    #                                          size=[x.shape[0], x.shape[1], 1, x.shape[3]])).float().to(x.device)
    #     x_adv = torch.clamp(x + pert, 0, 1)
    #     loss, is_cache = margin_loss(x_adv, y)
    #
    #     pbar = tqdm(range(self.attack_config["max_iter"]))
    #     for t in pbar:
    #         x_adv_candidate = x_adv.clone()
    #         for _ in range(self.attack_config["num_squares"]):
    #             # pert = x_adv - x
    #             pert = x_adv_candidate - x
    #             s = int(min(max(torch.sqrt(p_selection(t) * dim / x.shape[1]).round().item(), 1), x.shape[2] - 1))
    #             center_h = torch.randint(0, x.shape[2] - s, size=(1,)).to(x.device)
    #             center_w = torch.randint(0, x.shape[3] - s, size=(1,)).to(x.device)
    #             x_window = x[:, :, center_h:center_h + s, center_w:center_w + s]
    #             x_adv_window = x_adv_candidate[:, :, center_h:center_h + s, center_w:center_w + s]
    #
    #             while torch.sum(
    #                     torch.abs(
    #                         torch.clamp(
    #                             x_window + pert[:, :, center_h:center_h + s, center_w:center_w + s], 0, 1
    #                         ) -
    #                         x_adv_window)
    #                     < 10 ** -7) == x_adv.shape[1] * s * s:
    #                 pert[:, :, center_h:center_h + s, center_w:center_w + s] = torch.tensor(
    #                     np.random.choice([-self.attack_config["eps"], self.attack_config["eps"]], size=[x_adv.shape[1], 1, 1])).float().to(x_adv.device)
    #
    #             x_adv_candidate = torch.clamp(x + pert, 0, 1)
    #         new_loss, is_cache = margin_loss(x_adv_candidate, y)
    #         if is_cache[0]:
    #             continue
    #         if new_loss < loss:
    #             x_adv = x_adv_candidate.clone()
    #             loss = new_loss
    #         pbar.set_description(
    #             f"Step: {t} | True Label: {y} | Predicted Label: {torch.argmax(self._model.model(x_adv))} | Loss: {loss} | square_size: {s} | Cache Hits : {self._model.cache_hits}/{self._model.total}")
    #         if loss == 0:
    #             assert torch.max(torch.abs(x_adv - x)) <= self.attack_config["eps"] + 10 ** -4
    #             return x_adv
    #     return x

    def p_selection(self, step):
        step = int(step / self.attack_config["max_iter"] * 10000)
        if 10 < step <= 50:
            p = self.attack_config["p_init"] / 2
        elif 50 < step <= 200:
            p = self.attack_config["p_init"] / 4
        elif 200 < step <= 500:
            p = self.attack_config["p_init"] / 8
        elif 500 < step <= 1000:
            p = self.attack_config["p_init"] / 16
        elif 1000 < step <= 2000:
            p = self.attack_config["p_init"] / 32
        elif 2000 < step <= 4000:
            p = self.attack_config["p_init"] / 64
        elif 4000 < step <= 6000:
            p = self.attack_config["p_init"] / 128
        elif 6000 < step <= 8000:
            p = self.attack_config["p_init"] / 256
        elif 8000 < step <= 10000:
            p = self.attack_config["p_init"] / 512
        else:
            p = self.attack_config["p_init"]
        return p

    def margin_loss(self, x, y):
        logits, is_cache = self.model(x)
        self.update_all_probabilities(logits, is_cache)
        probs = torch.softmax(logits, dim=1)
        top_2_probs, top_2_classes = torch.topk(probs, 2)
        if top_2_classes[:, 0] != y:
            print("Margin Loss is 0 because the top predicted class is not true class")
            logging.info(
                "Margin Loss is 0 because the top predicted class is not true class"
            )
            return 0, is_cache
        else:
            print(
                "Margin Loss is calculated as ", top_2_probs[:, 0] - top_2_probs[:, 1]
            )
            return top_2_probs[:, 0] - top_2_probs[:, 1], is_cache

    def attack_untargeted(self, x, y):
        print("Untargeted Square Attack Started...")
        print("TRUE LABEL : ", y)

        dim = torch.prod(torch.tensor(x.shape[1:]))

        # Initialize adversarial example
        pert = (
            torch.tensor(
                np.random.choice(
                    [-self.attack_config["eps"], self.attack_config["eps"]],
                    size=[x.shape[0], x.shape[1], 1, x.shape[3]],
                )
            )
            .float()
            .to(x.device)
        )
        print("Initial image shape=", x.shape)
        print("Initial perturbation shape=", pert.shape)
        x_adv = torch.clamp(x + pert, 0, 1)

        loss, is_cache = self.margin_loss(x_adv, y)

        print("Begin Estimate Initial NUM_SQUARES")

        if self.attack_config["adaptive"]["bs_num_squares"]:
            ns = self.binary_search_num_squares(x, x_adv)
        else:
            ns = 1

        print("End Estimate. Optimal NUM_SQUARES = ", ns)

        print("Begin Estimate Initial MIN_SQUARE_SIZE")
        if self.attack_config["adaptive"]["bs_min_square_size"]:
            min_s = self.binary_search_min_square_size(x, x_adv, ns)
        else:
            min_s = 1
        print("End Estimate. Optimal MIN_SQUARE_SIZE = ", min_s)

        pbar = tqdm(range(self.attack_config["max_iter"]))
        step_attempts = 0

        for t in pbar:

            self.plotIndex.append(self.currentIndex)
            self.index_label.append(f"Iteration {t+1}")
            print(f"Begin Iteration {t+1}")

            # x_adv_candidate = x_adv.clone()
            s = int(
                min(
                    max(
                        torch.sqrt(self.p_selection(t) * dim / x.shape[1])
                        .round()
                        .item(),
                        1,
                    ),
                    x.shape[2] - 1,
                )
            )
            if self.attack_config["adaptive"]["bs_min_square_size"]:
                s = max(s, min_s)

            print(f"Min Square Size updated after p_selection: {s}")

            x_adv_candidate = self.add_squares(x, x_adv, s, ns)

            step_attempts += 1
            new_loss, is_cache = self.margin_loss(x_adv_candidate, y)
            if (
                is_cache[0]
                and step_attempts < self.attack_config["adaptive"]["max_step_attempts"]
            ):

                # pbar.set_description(
                #     f"Step: {t} | True Label: {y} | Predicted Label: {torch.argmax(self._model.model(x_adv))} | Loss: {loss} | square_size: {s} | Cache Hits : {self._model.cache_hits}/{self._model.total}"
                # )
                print(
                    f"Step: {t} | True Label: {y} | Predicted Label: {torch.argmax(self._model.model(x_adv))} | Loss: {loss} | square_size: {s} | Cache Hits : {self._model.cache_hits}/{self._model.total}"
                )
                continue
            elif (
                is_cache[0]
                and step_attempts >= self.attack_config["adaptive"]["max_step_attempts"]
            ):
                self.plotIndex.append(self.currentIndex - 1)
                self.index_label.append(f"Step movement failure")
                self.writeto_csv()
                self.plot_iterative_probs_each_image()
                print("Step movement failure.")
                self.end("Step movement failure.")

            step_attempts = 0
            if new_loss < loss:
                print("Succesful Iteration: found new potential ADV Image sample")
                x_adv = x_adv_candidate.clone()
                loss = new_loss
            # pbar.set_description(
            #     f"Step: {t} | True Label: {y} | Predicted Label: {torch.argmax(self._model.model(x_adv))} | Loss: {loss} | square_size: {s} | Cache Hits : {self._model.cache_hits}/{self._model.total}"
            # )
            print(
                f"Iteration Step: {t} | True Label: {y} | Predicted Label: {torch.argmax(self._model.model(x_adv))} | Loss: {loss} | square_size: {s} | Cache Hits : {self._model.cache_hits}/{self._model.total}"
            )
            if loss == 0:
                assert (
                    torch.max(torch.abs(x_adv - x))
                    <= self.attack_config["eps"] + 10**-4
                )
                self.plotIndex.append(self.currentIndex - 1)
                self.index_label.append(f"Adv Img Generated")
                self.writeto_csv()
                self.plot_iterative_probs_each_image()
                print("Adversarial Image Generated.")
                return x_adv

        self.plotIndex.append(self.currentIndex - 1)
        self.index_label.append(f"Max Iterations completed")
        self.writeto_csv()
        self.plot_iterative_probs_each_image()
        return x

    def add_squares(self, x, x_adv, s, num_squares):
        x_adv_candidate = x_adv.clone()
        for _ in range(num_squares):
            pert = x_adv_candidate - x

            center_h = torch.randint(0, x.shape[2] - s, size=(1,)).to(x.device)
            center_w = torch.randint(0, x.shape[3] - s, size=(1,)).to(x.device)
            x_window = x[:, :, center_h : center_h + s, center_w : center_w + s]
            x_adv_window = x_adv_candidate[
                :, :, center_h : center_h + s, center_w : center_w + s
            ]

            while (
                torch.sum(
                    torch.abs(
                        torch.clamp(
                            x_window
                            + pert[
                                :, :, center_h : center_h + s, center_w : center_w + s
                            ],
                            0,
                            1,
                        )
                        - x_adv_window
                    )
                    < 10**-7
                )
                == x.shape[1] * s * s
            ):

                pert[:, :, center_h : center_h + s, center_w : center_w + s] = (
                    torch.tensor(
                        np.random.choice(
                            [-self.attack_config["eps"], self.attack_config["eps"]],
                            size=[x.shape[1], 1, 1],
                        )
                    )
                    .float()
                    .to(x.device)
                )
            x_adv_candidate = torch.clamp(x + pert, 0, 1)
        return x_adv_candidate

    def binary_search_num_squares(self, x, x_adv):

        self.plotIndex.append(self.currentIndex)
        self.index_label.append("Initial_Num_Squares")

        dim = torch.prod(torch.tensor(x.shape[1:]))
        lower = self.attack_config["adaptive"]["bs_num_squares_lower"]
        upper = self.attack_config["adaptive"]["bs_num_squares_upper"]
        ns = upper
        for _ in range(self.attack_config["adaptive"]["bs_num_squares_steps"]):
            mid = (lower + upper) / 2
            cache_hits = 0
            for _ in range(
                self.attack_config["adaptive"]["bs_num_squares_sample_size"]
            ):
                s = int(
                    min(
                        max(
                            torch.sqrt(self.p_selection(0) * dim / x.shape[1])
                            .round()
                            .item(),
                            1,
                        ),
                        x.shape[2] - 1,
                    )
                )
                noisy_img = self.add_squares(x, x_adv, s, int(mid))
                probs, is_cache = self.model(noisy_img)
                self.update_all_probabilities(probs, is_cache)

                if is_cache[0]:
                    cache_hits += 1
            if (
                cache_hits
                / self.attack_config["adaptive"]["bs_num_squares_sample_size"]
                <= self.attack_config["adaptive"]["bs_num_squares_hit_rate"]
            ):
                ns = mid
                upper = mid
            else:
                lower = mid
            print(
                f"Num Squares : {ns:.6f} | Cache Hits : {cache_hits}/{self.attack_config['adaptive']['bs_num_squares_sample_size']}"
            )

        return int(ns)

    def binary_search_min_square_size(self, x, x_adv, num_squares):

        self.plotIndex.append(self.currentIndex)
        self.index_label.append("Initial_min_sq_size")

        lower = self.attack_config["adaptive"]["bs_min_square_size_lower"]
        upper = self.attack_config["adaptive"]["bs_min_square_size_upper"]
        min_ss = upper
        for _ in range(self.attack_config["adaptive"]["bs_min_square_size_steps"]):
            mid = (lower + upper) / 2
            cache_hits = 0
            for _ in range(
                self.attack_config["adaptive"]["bs_min_square_size_sample_size"]
            ):
                noisy_img = self.add_squares(x, x_adv, int(mid), num_squares)
                probs, is_cache = self.model(noisy_img)
                self.update_all_probabilities(probs, is_cache)
                if is_cache[0]:
                    cache_hits += 1
            if (
                cache_hits
                / self.attack_config["adaptive"]["bs_min_square_size_sample_size"]
                <= self.attack_config["adaptive"]["bs_min_square_size_hit_rate"]
            ):
                min_ss = mid
                upper = mid
            else:
                lower = mid
            print(
                f"Min Square Size : {min_ss:.6f} | Cache Hits : {cache_hits}/{self.attack_config['adaptive']['bs_min_square_size_sample_size']}"
            )
        return int(min_ss)

    def update_all_probabilities(self, curr_probs, is_cache):

        # print("SQUARE logits")
        # print("SQUARE logits, is_cache=", is_cache[0])

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
        self.cache_values_all.append(is_cache[0])
        self.currentIndex += 1

    def writeto_csv(self):

        directory = "csvfiles"
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"testfile{self.image_number}.csv")
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "CurrentIndex",
                    "IndexLabel",
                    "Square_Logits_True/False",
                    "SortedList_First_Element",
                    "SortedList_Second_Element",
                    "PredictedLabel_first",
                    "Predicted_Label_second",
                    "Margin_Loss",
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
            if idx < 3 or idx == len(self.plotIndex) - 1:
                plt.axvline(x=highlight_x + 1, color="g", linestyle="--", linewidth=1)
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
        plt.title(" Square Plot ")
        adjusted_plotIndex = np.array([i + 1 for i in self.plotIndex])

        # xticks = list(plt.xticks()[0])
        # max_x_tick = max(xticks)
        # min_x_tick = max( min(xticks), 0)

        # tick_range = np.arange(min_x_tick, max_x_tick + 25, 25)
        # xticks_combined = np.unique(np.concatenate([tick_range, adjusted_plotIndex]))

        # xticks_combined = xticks_combined[xticks_combined != 0]
        # if 1 not in xticks_combined:
        #     xticks_combined = np.append(xticks_combined, 1)
        # xticks_combined = np.sort(xticks_combined)
        # plt.xticks(xticks_combined, fontsize=6)

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
